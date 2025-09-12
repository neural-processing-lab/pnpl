import os
import warnings
from torch.utils.data import Dataset
import torch
import numpy as np
import mne_bids
import h5py


class HDF5Dataset(Dataset):
    """Torch Dataset for Serialized hdf5 files.
    Please implement get_phonemes method in the subclass.
    """

    def __init__(
        self,
        # use /data/engs-pnpl/datasets/Sherlock1/derivatives/serialized/default
        data_path: str,
        preprocessing_name: str | None = None,
        tmin: float = -0.2,
        tmax: float = 0.6,
        sfreq: float | None = None,
        subjects: list[str] = [],
        sessions: list[str | None] = [None],
        tasks: list[str | None] = [None],
        runs: list[str | None] = [None],
        include_info: bool = False,
        standardize: bool = True,
        clipping_factor: float | None = None,
        channel_means: np.ndarray | None = None,
        channel_stds: np.ndarray | None = None,
    ):
        """
        data_path: path to serialized dataset.
        include: Subjects to load (e.g. ['010002', '010047']). If empty load all
        partition: train, val, or test

        #included_files_dict: dictionary to indicate which files to load.
            Keys are subjects and values are sessions. If empty, load all.

        standardize: If True, standardize the data. Calculate means and stds if not provided.
        """

        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")
        if (clipping_factor is not None and not standardize):
            raise ValueError("Clipping without standardizing not implemented.")

        self.data_path = data_path
        self.preprocessing_name = preprocessing_name
        self.tmin = tmin
        self.tmax = tmax
        self.standardize = standardize
        self.clipping_factor = clipping_factor
        self.include_info = include_info
        self.subjects = subjects
        self.sessions = sessions
        self.tasks = tasks
        self.runs = runs
        # self.samples contains subject, session, task, run, onset, label where label is the full phoneme label (e.g. "aa_S") where the first part is the phoneme and the second part is the location of the phoneme in the word
        self.samples = []
        for subject in subjects:
            for session in sessions:
                for task in tasks:
                    for run in runs:
                        try:
                            self._collect_samples(subject, session, task, run)
                        except FileNotFoundError:
                            warnings.warn(
                                f"File not found for {subject}, {session}, {task}, {run}. Skipping")

        if (len(self.samples) == 0):
            raise ValueError("No samples found at events path.")

        if (sfreq is None):
            sfreq = self._get_sfreq()
        self.sfreq = sfreq
        self.points_per_sample = int((tmax - tmin) * sfreq)
        self.open_h5_datasets = {}
        self.phoneme_labels = self._get_unique_labels()
        self.phoneme_to_id = {label: i for i,
                              label in enumerate(self.phoneme_labels)}
        self.id_to_phoneme = self.phoneme_labels

        self.labels = self.phoneme_labels
        self.label_to_id = self.phoneme_to_id

        if (self.standardize and channel_means is None and channel_stds is None):
            self._calculate_standardization_params()
        elif (self.standardize and (channel_means is not None and channel_stds is not None)):
            self.channel_means = channel_means
            self.channel_stds = channel_stds
            self.broadcasted_stds = np.tile(
                self.channel_stds, (self.points_per_sample, 1)).T
            self.broadcasted_means = np.tile(
                self.channel_means, (self.points_per_sample, 1)).T

    def _collect_samples(self, subject, session, task, run):
        if (session is None and task is None and run is None):
            labels, onsets = self.get_phonemes(subject)
        elif (run is None and task is None):
            labels, onsets = self.get_phonemes(
                subject, session)
        elif (run is None):
            labels, onsets = self.get_phonemes(
                subject, session, task)
        else:
            labels, onsets = self.get_phonemes(
                subject, session, task, run)
        for label, onset in zip(labels, onsets):
            self.samples.append(
                (subject, session, task, run, onset, label))

    def _calculate_standardization_params(self):
        n_samples = []
        means = []
        stds = []
        for subject in self.subjects:
            for session in self.sessions:
                for task in self.tasks:
                    hdf_dataset = h5py.File(
                        self._ids_to_h5_path(subject, session, task, self.runs[0]), "r")["data"]
                    info = hdf_dataset.attrs
                    n_samples.append(info["n_samples"])
                    means.append(info["channel_means"])
                    stds.append(info["channel_stds"])
        means = np.array(means)
        stds = np.array(stds)
        n_samples = np.array(n_samples)
        self.channel_stds, self.channel_means = self._accumulate_stds(
            means, stds, n_samples)
        self.broadcasted_stds = np.tile(
            self.channel_stds, (self.points_per_sample, 1)).T
        self.broadcasted_means = np.tile(
            self.channel_means, (self.points_per_sample, 1)).T

    def get_phonemes(self, subject, session, task, run):
        raise NotImplementedError("Please implement this method.")

    def _get_sfreq(self):
        h5_path = self._ids_to_h5_path(
            self.subjects[0], self.sessions[0], self.tasks[0], self.runs[0])
        h5_dataset = h5py.File(h5_path, "r")["data"]
        info = h5_dataset.attrs
        sfreq = info["sfreq"]
        return sfreq

    @staticmethod
    def _accumulate_stds(ch_means, ch_stds, n_samples):
        """
        ch_means: np.ndarray (n_groups, n_channels)
        ch_stds: np.ndarray (n_groups, n_channels)
        n_samples: np.ndarray (n_groups)
        """
        vars = np.array(ch_stds) ** 2
        means_total = np.average(ch_means, axis=0, weights=n_samples)
        sum_of_squares_within = np.sum(
            vars * np.tile(n_samples, (vars.shape[1], 1)).T, axis=0)
        sum_of_squares_between = np.sum((
            ch_means - np.tile(means_total, (ch_means.shape[0], 1)))**2 * np.tile(n_samples, (ch_means.shape[1], 1)).T, axis=0)
        sum_of_squares_total = sum_of_squares_within + sum_of_squares_between
        return np.sqrt(sum_of_squares_total / np.sum(n_samples)), means_total

    def _get_unique_labels(self):
        labels = set()
        for i in range(len(self)):
            labels.add(self.samples[i][5])
        labels = list(labels)
        labels.sort()
        return labels

    def _ids_to_h5_path(self, subject, session, task, run):
        bids_path = mne_bids.BIDSPath(
            subject=subject,
            session=session,
            task=task,
            run=run,
            processing=self.preprocessing_name,
            datatype="meg",
            root=self.data_path,
        )
        return str(bids_path.fpath) + ".h5"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # a little hacky because I want to keep full phoneme labels for sherlock1 for now
        if (idx >= len(self.samples)):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.samples)}")
        subject, session, task, run, onset, phoneme_full = self.samples[idx]
        phoneme = phoneme_full.split("_")[0]

        if (not ((subject, session, task, run) in self.open_h5_datasets)):
            h5_path = self._ids_to_h5_path(subject, session, task, run)
            h5_dataset = h5py.File(
                h5_path, "r")["data"]
            self.open_h5_datasets[(
                subject, session, task, run)] = h5_dataset
        else:
            h5_dataset = self.open_h5_datasets[(subject, session, task, run)]

        start = max(0, int((onset + self.tmin) * self.sfreq))
        end = start + self.points_per_sample
        data = h5_dataset[:, start:end]

        if (self.clipping_factor is not None):
            data = self._clip_sample(data, self.clipping_factor)

        if (self.standardize):
            data = (data - self.broadcasted_means) / self.broadcasted_stds

        phoneme_id = self.phoneme_to_id[phoneme]

        if (self.include_info):
            info = dict(h5_dataset.attrs)
            return [torch.tensor(data, dtype=torch.float32), torch.tensor(phoneme_id), info]
        return [torch.tensor(data, dtype=torch.float32), torch.tensor(phoneme_id)]

    def _clip_sample(self, sample, factor):
        sample = np.clip(sample, (-factor * self.broadcasted_stds) +
                         self.broadcasted_means, (factor * self.broadcasted_stds) + self.broadcasted_means)
        return sample


def test_accumulate_stds():
    groups = [np.random.randn(2, 10), np.random.randn(
        2, 20), np.random.randn(2, 15), np.random.randn(2, 100)]
    full = np.concatenate(groups, axis=1)
    full_stds = np.std(full, axis=1)
    full_means = np.mean(full, axis=1)
    n_samples = [group.shape[1] for group in groups]
    group_means = np.array([group.mean(axis=1) for group in groups])
    group_stds = np.array([group.std(axis=1) for group in groups])
    stds, means = HDF5Dataset._accumulate_stds(
        group_means, group_stds, n_samples)
    assert np.allclose(full_means, means)
    assert np.allclose(full_stds, stds)
    print("Test passed.")
