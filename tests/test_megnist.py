"""Tests for the MegNIST dataset loader."""

import h5py
import numpy as np
import torch

from pnpl.datasets import MegNIST
from pnpl.tasks import DigitClassification


def _make_mock_megnist(root, partition="train"):
    """Create a tiny MegNIST-style HDF5 file for testing."""
    path = root / "derivatives" / "serialised"
    path.mkdir(parents=True)

    filename = path / f"{partition}.h5"

    rng = np.random.default_rng(42)
    data = rng.normal(size=(20, 306, 250)).astype(np.float32)
    labels = np.tile(np.arange(10), 2).astype(np.int64)
    times = np.arange(250, dtype=np.float32) / 250.0 - 0.048

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)
        f.create_dataset("times", data=times)
        f.create_dataset("sensor_xyz", data=np.zeros((306, 3), dtype=np.float32))

        string_dtype = h5py.string_dtype("utf-8")
        f.create_dataset(
            "channel_names",
            data=np.asarray([f"MEG{i:03d}" for i in range(306)], dtype=object),
            dtype=string_dtype,
        )
        f.create_dataset(
            "channel_types",
            data=np.asarray(["meg"] * 306, dtype=object),
            dtype=string_dtype,
        )

        f.attrs["sample_frequency"] = 250.0

    return filename


def test_public_imports():
    assert MegNIST is not None
    assert DigitClassification is not None


def test_digit_classification():
    task = DigitClassification()

    assert task.n_classes == 10
    assert task.classes[0] == "zero"
    assert task.classes[9] == "nine"
    assert task.label_info["label_to_id"]["zero"] == 0
    assert task.label_info["label_to_id"]["nine"] == 9


def test_megnist_loads_epoched_data(tmp_path):
    _make_mock_megnist(tmp_path)

    dataset = MegNIST(
        data_path=str(tmp_path),
        partition="train",
        standardize=False,
        download=False,
    )

    assert len(dataset) == 20
    assert dataset.sfreq == 250.0

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (306, 250)
    assert y.item() == 0


def test_validation_alias(tmp_path):
    _make_mock_megnist(tmp_path, partition="val")

    dataset = MegNIST(
        data_path=str(tmp_path),
        partition="validation",
        standardize=False,
        download=False,
    )

    assert dataset.partition == "val"
    assert len(dataset) == 20


def test_include_info(tmp_path):
    _make_mock_megnist(tmp_path)

    dataset = MegNIST(
        data_path=str(tmp_path),
        partition="train",
        standardize=False,
        include_info=True,
        download=False,
    )

    x, y, info = dataset[0]

    assert x.shape == (306, 250)
    assert y.item() == 0
    assert info["dataset"] == "megnist"
    assert info["partition"] == "train"
    assert info["trial_idx"] == 0