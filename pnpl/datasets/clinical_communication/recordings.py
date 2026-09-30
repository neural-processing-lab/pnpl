"""Pinned downloads and bounded, lazy caches for clinical word windows."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .manifest import asset_path, load, sha256

PIPELINE = "clinical-v1: .1-40Hz;50Hz;recording-RobustScaler;float64-cache;float32-window;baseline25;clip5"


def word_window(data, onset):
    start = round(float(onset) * 50)
    if start < 0:
        raise ValueError("Negative clinical word-window start")
    x = torch.from_numpy(np.array(data[:, start:start + 150], dtype=np.float32, copy=True))
    if x.shape != (306, 150):
        raise ValueError(f"Incomplete clinical word window: {tuple(x.shape)}")
    x -= x[:, :25].mean(dim=-1, keepdim=True)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite clinical word window")
    return x.clamp(-5, 5)


def _strings(value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in value]


class RecordingStore:
    def __init__(self, root, *, cache_path=None, download=True):
        self.root = Path(root).expanduser()
        self.cache = Path(cache_path).expanduser() if cache_path else self.root / ".clinical_cache"
        self.download = download
        self._arrays = OrderedDict()
        self._verified = {}
        self._pid = os.getpid()

    def _source(self, relative, checksum):
        path = self.root / relative
        spec = load("downloads.json")[relative]
        if not path.is_file():
            if not self.download:
                raise FileNotFoundError(f"Missing clinical source: {path}; enable download or provide the pinned recording")
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=spec["repo"], repo_type="dataset",
                revision=spec["revision"], filename=relative, local_dir=str(self.root)))
        if path.stat().st_size != spec["size"] or sha256(path) != checksum:
            raise ValueError(f"Clinical source checksum/size mismatch: {path}")
        return path

    def _identity(self, record):
        import mne
        import sklearn
        return dict(pipeline=PIPELINE, neural=record["neural"],
                    source_sha256=load("downloads.json")[record["neural"]]["sha256"],
                    events_sha256=record["events_sha256"], origin=record["origin"],
                    mne=mne.__version__, sklearn=sklearn.__version__,
                    numpy=np.__version__, torch=torch.__version__)

    def _prepare(self, record, folder, identity):
        import h5py
        import mne
        from sklearn.preprocessing import RobustScaler

        self._source(record["events"], record["events_sha256"])
        path = self._source(record["neural"], identity["source_sha256"])
        with h5py.File(path, "r") as f:
            names = _strings(f.attrs["channel_names"])
            kinds = _strings(f.attrs["channel_types"])
            rate = float(f.attrs["sample_frequency"])
            times = np.asarray(f["times"])
            if len(names) != 306 or rate != record["rate"] or float(times[0]) != record["origin"]:
                raise ValueError("Clinical recording channels, sample rate or time origin changed")
            tolerance = max(1e-6, float(np.finfo(times.dtype).eps) * max(1., float(abs(times).max())))
            expected = times[0] + np.arange(len(times)) / rate
            if not np.allclose(times, expected, atol=tolerance, rtol=0):
                raise ValueError("Clinical recording has a nonuniform time axis")
            raw = mne.io.RawArray(np.asarray(f["data"], dtype=np.float64),
                mne.create_info(names, rate, kinds), verbose="ERROR")
        raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[]))
        layout = mne.channels.read_layout("Vectorview-all")
        lookup = {n.replace(" ", ""): i for i, n in enumerate(layout.names)}
        xy = layout.pos[[lookup[n] for n in raw.ch_names], :2]
        positions = ((xy - xy.min(0)) / np.ptp(xy, axis=0)).astype("float32")
        if not np.array_equal(positions, np.load(asset_path("positions.npy"))):
            raise ValueError("Clinical recording channel order differs from benchmark")
        raw.filter(.1, 40., n_jobs=1, verbose="ERROR").resample(50., n_jobs=1, verbose="ERROR")
        scaled = RobustScaler().fit_transform(raw.get_data().T).T
        raw.close()
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite preprocessed clinical recording")
        # Unique staging names and an atomic receipt prevent partial-cache reuse.
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            tmp = Path(temporary)
            np.save(tmp / "continuous.npy", scaled)
            receipt = dict(identity=identity, shape=list(scaled.shape), dtype=str(scaled.dtype),
                           sha256=sha256(tmp / "continuous.npy"))
            (tmp / "receipt.json").write_text(json.dumps(receipt))
            os.replace(tmp / "continuous.npy", folder / "continuous.npy")
            os.replace(tmp / "receipt.json", folder / "receipt.json")

    def array(self, record):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        key = record["neural"]
        if key in self._arrays:
            self._arrays.move_to_end(key)
            return self._arrays[key]
        identity = self._identity(record)
        folder = self.cache / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt_path = folder / "receipt.json"
        if not receipt_path.exists():
            self._prepare(record, folder, identity)
        receipt = json.loads(receipt_path.read_text())
        path = folder / "continuous.npy"
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                 receipt["sha256"])
        if receipt["identity"] != identity:
            raise ValueError(f"Clinical cache identity mismatch: {folder}")
        if self._verified.get(str(path)) != stamp:
            if sha256(path) != receipt["sha256"]:
                raise ValueError(f"Clinical cache checksum mismatch: {folder}")
            self._verified[str(path)] = stamp
        array = np.load(path, mmap_mode="r")
        if list(array.shape) != receipt["shape"] or array.dtype != np.float64 or array.shape[0] != 306:
            raise ValueError("Clinical cache shape/dtype mismatch")
        self._arrays[key] = array
        while len(self._arrays) > 2:
            _, old = self._arrays.popitem(last=False)
            old._mmap.close()
        return array

    def window(self, record, onset):
        return word_window(self.array(record), onset)

    def close(self):
        for array in self._arrays.values():
            array._mmap.close()
        self._arrays.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = OrderedDict()
        return state
