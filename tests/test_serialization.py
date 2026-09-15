"""fif_to_h5 and epochs_to_h5 store MEG by default and other channel types on request."""

import h5py
import mne
import numpy as np
import pytest

from pnpl.preprocessing.serialization import epochs_to_h5, fif_to_h5

NAMES = ["MEG0111", "MEG0112", "EEG001", "EEG002", "EOG061"]
TYPES = ["mag", "grad", "eeg", "eeg", "eog"]


def _raw():
    info = mne.create_info(NAMES, 250.0, ch_types=TYPES)
    return mne.io.RawArray(np.random.RandomState(0).standard_normal((len(NAMES), 500)) * 1e-12, info, verbose=False)


@pytest.mark.parametrize(
    "picks, expected",
    [
        ("meg", NAMES[:2]),
        (["meg", "eeg"], NAMES[:4]),
        ("eeg", NAMES[2:4]),
        ("all", NAMES),
    ],
)
def test_fif_to_h5_picks(tmp_path, picks, expected):
    path = fif_to_h5(_raw(), str(tmp_path / "raw.h5"), picks=picks)
    with h5py.File(path, "r") as f:
        assert f.attrs["channel_names"].split(", ") == expected
        assert f["data"].shape == (len(expected), 500)


def test_fif_to_h5_default_keeps_meg_only(tmp_path):
    path = fif_to_h5(_raw(), str(tmp_path / "raw.h5"))
    with h5py.File(path, "r") as f:
        assert f.attrs["channel_types"].split(", ") == ["mag", "grad"]


@pytest.mark.parametrize("picks, expected", [("meg", NAMES[:2]), (["meg", "eeg"], NAMES[:4])])
def test_epochs_to_h5_data_matches_channel_list(tmp_path, picks, expected):
    events = np.array([[100, 0, 1], [300, 0, 2]])
    epochs = mne.Epochs(_raw(), events, tmin=0.0, tmax=0.2, baseline=None, preload=True, verbose=False)
    path = epochs_to_h5(epochs, str(tmp_path / "epo.h5"), picks=picks)
    with h5py.File(path, "r") as f:
        assert [n.decode() for n in f["channel_names"][()]] == expected
        assert f["data"].shape[1] == len(expected)
        assert f["sensor_xyz"].shape == (len(expected), 3)
