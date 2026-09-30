"""Frozen occurrence, split, k and preprocessing contracts (no network needed)."""
import hashlib
import json
import pickle

import numpy as np
import pytest
import torch

from pnpl.datasets import ClinicalCommunication
from pnpl.datasets.clinical_communication.manifest import load
from pnpl.datasets.clinical_communication.recordings import RecordingStore, word_window


def dataset(tmp_path, split="test", **kwargs):
    return ClinicalCommunication(tmp_path, split, download=False, **kwargs)


def test_original_splits_and_development_source(tmp_path):
    sets = {}
    for split, count, recordings in [("train", 595598, 121), ("val", 51126, 10),
                                      ("test", 100, 31), ("dev", 50, 10)]:
        ds = dataset(tmp_path, split)
        assert len(ds) == count and len(ds.recordings) == recordings
        sets[split] = {r["neural"] for r in ds.recordings}
        assert {r["subject"] for r in ds.recordings} == {"0"}
    assert sets["train"].isdisjoint(sets["val"] | sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    assert sets["dev"] == sets["val"]
    assert {r["session"] for r in dataset(tmp_path, "val").recordings} == {"11"}


def test_frozen_sentence_counts_and_no_reused_donors(tmp_path):
    for split, count, positions in [("test", 100, 595), ("test", 200, 1275), ("dev", 100, 307)]:
        ds = dataset(tmp_path, split, test_sentences=count)
        donors = [(d["neural"], d["event_index"]) for i in range(len(ds))
                  for group in ds.metadata(i)["occurrences"] for d in group]
        assert len(donors) == 5 * positions
        assert len(set(donors)) == len(donors)
        assert len(ds.candidate_words) == 92
    ds = dataset(tmp_path)
    assert ds.metadata(0)["sentence"] == "Can you help me?"
    assert ds.metadata(0)["words"] == ["can", "you", "help", "me"]
    assert ds.metadata(-1) == ds.metadata(99)
    with pytest.raises(IndexError):
        ds.metadata(100)


def test_nested_k_preserves_every_ordered_occurrence(tmp_path):
    full = dataset(tmp_path)
    for k in range(1, 6):
        small = dataset(tmp_path, k=k)
        for i in range(100):
            a, b = small.metadata(i), full.metadata(i)
            assert a["words"] == b["words"]
            assert a["occurrences"] == [donors[:k] for donors in b["occurrences"]]
    expanded = dataset(tmp_path, test_sentences=200)
    assert all(full.metadata(i) == expanded.metadata(i) for i in range(100))


def test_dev_uses_original_validation_groups(tmp_path):
    ds = dataset(tmp_path, "dev")
    groups = load("groups.json.gz")
    for g in groups["dev"]:
        assert g["indices"] == groups["val"][g["validation_group_index"]]["indices"]
    assert len(ds) == 50
    assert ds.metadata(0)["sentence"] == "Help me with this."


def test_natural_sentence_and_frozen_group_views(tmp_path):
    for split, n_sentences, n_groups in [("train", 45072, 111917), ("val", 3830, 6845)]:
        sentences = dataset(tmp_path, split, unit="sentence")
        groups = dataset(tmp_path, split, unit="group", k=3)
        assert len(sentences) == n_sentences
        assert len(groups) == n_groups
        assert len(groups.metadata(0)["occurrences"][0]) == 3
        assert all(len(g) == 1 for g in sentences.metadata(0)["occurrences"])


@pytest.mark.parametrize("bad", [0, 6, -1, 2.5, True, "3"])
def test_invalid_k(tmp_path, bad):
    with pytest.raises(ValueError, match="k must"):
        dataset(tmp_path, k=bad)


def test_invalid_options_and_aliases(tmp_path):
    for kwargs in [dict(test_sentences=50), dict(unit="average")]:
        with pytest.raises(ValueError):
            dataset(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        dataset(tmp_path, "unknown")
    assert dataset(tmp_path, "development").partition == "dev"
    assert dataset(tmp_path, "validation").partition == "val"


def test_getitem_keeps_occurrences_separate_and_is_worker_serializable(tmp_path, monkeypatch):
    ds = dataset(tmp_path, k=3)
    def fake_window(record, onset):
        return torch.full((306, 150), float(onset))
    monkeypatch.setattr(ds.store, "window", fake_window)
    row = ds[0]
    assert row["meg"].shape == (4, 3, 306, 150)
    for word, donors in enumerate(row["occurrences"]):
        for member, donor in enumerate(donors):
            assert row["meg"][word, member, 0, 0].item() == pytest.approx(donor["onset"])
    # Public metadata is safe for callers to mutate.
    row["occurrences"][0][0]["event_index"] = -1
    assert ds.metadata(0)["occurrences"][0][0]["event_index"] != -1
    fresh = dataset(tmp_path, k=2)
    restored = pickle.loads(pickle.dumps(fresh))
    assert restored.metadata(0) == fresh.metadata(0)


def test_missing_sources_and_wrong_checksum_fail_closed(tmp_path):
    store = RecordingStore(tmp_path, download=False)
    r = load("recordings.json")[0]
    with pytest.raises(FileNotFoundError, match="Missing clinical source"):
        store._source(r["events"], r["events_sha256"])
    p = tmp_path / r["events"]
    p.parent.mkdir(parents=True)
    p.write_text("wrong annotations")
    with pytest.raises(ValueError, match="checksum"):
        store._source(r["events"], r["events_sha256"])


def test_window_order_rounding_baseline_and_clamp():
    rng = np.random.default_rng(1)
    signal = rng.normal(size=(306, 400)) * 8
    actual = word_window(signal, .011)
    reference = torch.tensor(signal[:, 1:151], dtype=torch.float32)
    reference -= reference[:, :25].mean(-1, keepdim=True)
    torch.testing.assert_close(actual, reference.clamp(-5, 5), rtol=0, atol=0)
    with pytest.raises(ValueError):
        word_window(signal, -1)
    with pytest.raises(ValueError):
        word_window(signal, 7)


def test_h5_preprocessing_matches_reference_and_cache_reloads(tmp_path, monkeypatch):
    import h5py
    import mne
    from sklearn.preprocessing import RobustScaler
    from pnpl.datasets.clinical_communication.manifest import DATA

    r = dict(load("recordings.json")[0])
    names = [s.replace(" ", "") for s in mne.channels.read_layout("Vectorview-all").names]
    # Use benchmark channel order, which need not be layout order.
    # positions.npy identifies each ordered channel without assuming name sorting.
    layout = mne.channels.read_layout("Vectorview-all")
    xy = layout.pos[:, :2]
    pos = ((xy - xy.min(0)) / np.ptp(xy, axis=0)).astype("float32")
    canonical = np.load(DATA / "positions.npy")
    indices = [np.where(np.all(pos == p, axis=1))[0][0] for p in canonical]
    names = [names[i] for i in indices]
    kinds = ["mag" if n.endswith("1") else "grad" for n in names]
    signal = np.random.default_rng(7).normal(size=(306, 2500))
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as f:
        f["data"] = signal
        f["times"] = np.arange(2500) / 250
        f.attrs["channel_names"] = names
        f.attrs["channel_types"] = kinds
        f.attrs["sample_frequency"] = 250.
    r.update(rate=250., origin=0.)
    store = RecordingStore(tmp_path, download=False)
    monkeypatch.setattr(store, "_source", lambda *a: path)
    x = store.window(r, .5)
    raw = mne.io.RawArray(signal, mne.create_info(names, 250., kinds), verbose="ERROR")
    raw.filter(.1, 40., verbose="ERROR").resample(50., verbose="ERROR")
    reference = RobustScaler().fit_transform(raw.get_data().T).T
    torch.testing.assert_close(x, word_window(reference, .5), rtol=0, atol=0)
    store.close()
    # A complete cache can be read without the source H5 or network.
    path.unlink()
    second = RecordingStore(tmp_path, download=False)
    torch.testing.assert_close(second.window(r, .5), x, rtol=0, atol=0)
    second.close()
    continuous = next((tmp_path / ".clinical_cache").glob("*/continuous.npy"))
    with continuous.open("r+b") as stream:
        stream.seek(-8, 2)
        stream.write(b"corrupt!")
    with pytest.raises(ValueError, match="cache checksum"):
        RecordingStore(tmp_path, download=False).window(r, .5)

# Golden hashes calculated from the original occurrence CSV and development
# validation-group indices, independently of the bundled PNPL manifests.
GOLDEN_DONORS = {'100': '4e37fddd7e6dfcc8f5e725af0ac6841d62b0e141bac12e1a80a074e2a5eed01e', '200': '571ec99de61245c1efeaf4d82e042e08c692072ca5f9f9e09febc51d6a5a9f5c', 'dev': '49deaca49868316ce0d7f772e62c435fa29ce53b1e7181aaba47e605fa505ba0'}

def test_original_experiment_occurrence_digests(tmp_path):
    for key, expected in GOLDEN_DONORS.items():
        ds = dataset(tmp_path, "dev" if key == "dev" else "test",
                     test_sentences=100 if key == "dev" else int(key))
        donors = []
        for i in range(len(ds)):
            row = ds.metadata(i)
            for position, (word, group) in enumerate(zip(row["words"], row["occurrences"]), 1):
                donors.extend([row["sentence_id"], position, word, d["neural"], d["event_index"]]
                              for d in group)
        digest = hashlib.sha256(json.dumps(donors, separators=(",", ":")).encode()).hexdigest()
        assert digest == expected
