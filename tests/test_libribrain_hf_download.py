import os
import io
import csv
import tempfile
from pathlib import Path

import numpy as np
import h5py
import pytest

from pnpl.datasets.libribrain2025.speech_dataset import LibriBrainSpeech
import pnpl.datasets.libribrain2025.base as lbbase


def _write_h5(path: str, channels: int = 4, samples: int = 1000, sfreq: float = 250.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        data = np.random.randn(channels, samples).astype("float32")
        dset = f.create_dataset("data", data=data)
        f.attrs["sample_frequency"] = sfreq


def _write_events_tsv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["kind", "timemeg", "duration"])  # header
        writer.writerows(rows)


@pytest.mark.parametrize("stride", [None, 50])
def test_libribrain_hf_download_and_sample_count(monkeypatch, stride):
    # Prepare a fake hf_hub_download that writes the requested file locally
    def fake_hf_hub_download(repo_id, repo_type, filename, local_dir, token=None):
        # filename is a relative posix path; write the appropriate file content
        full = Path(local_dir) / filename
        if str(full).endswith(".h5"):
            _write_h5(str(full), channels=3, samples=1000, sfreq=250.0)
        elif str(full).endswith("_events.tsv"):
            # Create alternating silence/word spans to ensure both labels exist
            # times in seconds
            rows = [
                ("silence", 0.0, 1.0),
                ("word", 1.0, 1.0),
                ("silence", 2.0, 1.0),
                ("word", 3.0, 1.0),
            ]
            _write_events_tsv(str(full), rows)
        else:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            Path(full).write_text("")
        return str(full)

    # Patch the exact function the dataset code calls
    monkeypatch.setattr(lbbase, "hf_hub_download", fake_hf_hub_download, raising=True)

    with tempfile.TemporaryDirectory() as tmp:
        # Choose a single run key
        include_run_keys = [["0", "1", "Sherlock1", "1"]]
        ds = LibriBrainSpeech(
            data_path=tmp,
            preprocessing_str="bads+headpos+sss+notch+bp+ds",
            include_run_keys=include_run_keys,
            exclude_run_keys=[],
            exclude_tasks=[],
            tmin=0.0,
            tmax=0.2,
            standardize=True,
            include_info=True,
            stride=stride,
            download=True,
        )

        # Verify files were created by "download"
        base = Path(tmp) / "Sherlock1" / "derivatives"
        h5s = list(base.glob("serialised/*.h5"))
        events = list(base.glob("events/*_events.tsv"))
        assert h5s and events

        # Expected number of samples: total duration is 4s, sfreq=250, window 0.2s
        # time_window_samples = 50. For stride=None, stride=time_window_samples; for stride=50, same.
        # speech_labels length ~= max end time in events (here 4s * 250 = 1000)
        time_window_samples = int((ds.tmax - ds.tmin) * ds.sfreq)
        expected = 1000 // (stride or time_window_samples)
        assert len(ds) == expected

        # One item returns (data, label, info) with correct shapes/types
        x, y, info = ds[0]
        assert x.shape[0] == 3
        assert x.shape[1] == time_window_samples
        assert y.numel() == 50 or y.numel() == time_window_samples  # segment labels
        assert info["dataset"] == "libribrain2025"
