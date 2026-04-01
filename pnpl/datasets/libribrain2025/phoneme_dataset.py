import os
import warnings
from pnpl.datasets.libribrain2025.constants import PHONATION_BY_PHONEME
import numpy as np
import torch
from pnpl.datasets.libribrain2025.base import LibriBrainBase
from .constants import PHONEME_LABELS_SORTED


class LibriBrainPhoneme(LibriBrainBase):

    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
        label_type: str = "phoneme",
        preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: list[str] = [],
        exclude_run_keys: list[str] = [],
        exclude_tasks: list[str] = [],
        standardize: bool = True,
        clipping_boundary: float | None = 10,
        channel_means: np.ndarray | None = None,
        channel_stds: np.ndarray | None = None,
        include_info: bool = False,
        preload_files: bool = False,
    ):
        """
        data_path: path to serialized dataset. 
        label_type: "phoneme" or "voicing". Voicing labels are derived from phoneme labels and indicate voiced and unvoiced phonemes. See https://en.wikipedia.org/wiki/Voice_(phonetics) for more information.
        preprocessing_str: Preprocessing string in the file name. Indicates Preprocessing steps applied to the data.
        tmin: start time of the sample in seconds in reference to the onset of the phoneme.
        tmax: end time of the sample in seconds in reference to the onset of the phoneme.
        standardize: Whether to standardize the data. Uses channel_means and channel_stds if provided. Otherwise it calculates mean and std for each channel of the dataset. 
        clipping_boundary: Min and max values to clip the data by.
        channel_means: Standardize using these channel means.
        channel_stds: Standardize using these channel stds.
        include_info: Whether to include info dict in the output. Info dict contains dataset name, subject, session, task, run, onset time of the sample, and full phoneme label that indicates if a phoneme is at the onset or offset of a word.
        preload_files: If true start parallel downloads of all sessions and runs into data_path. Otherwise it will download files as they are needed.

        returns Channels x Time
        """
        super().__init__(
            data_path=data_path,
            partition=partition,
            preprocessing_str=preprocessing_str,
            tmin=tmin,
            tmax=tmax,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            exclude_tasks=exclude_tasks,
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
            include_info=include_info,
            preload_files=preload_files
        )
        supported_label_types = ["phoneme", "voicing"]
        if (label_type not in supported_label_types):
            raise ValueError(
                f"Label type {label_type} not supported. Supported types: {supported_label_types}")
        self.label_type = label_type
        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")

        self.samples = []
        run_keys_missing = []
        self.run_keys = []
        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                labels, onsets = self.load_phonemes_from_tsv(
                    subject, session, task, run)
                for label, onset in zip(labels, onsets):
                    sample = (subject, session, task, run, onset, label)
                    self.samples.append(sample)
                self.run_keys.append(run_key)
            except FileNotFoundError:
                run_keys_missing.append(run_key)
                warnings.warn(
                    f"File not found for run key {run_key}. Skipping")
                continue

        if len(run_keys_missing) > 0:
            warnings.warn(
                f"Run keys {run_keys_missing} not found in dataset. Present run keys: {self.run_keys}")

        if len(self.samples) == 0:
            raise ValueError("No samples found.")

        self.phonemes_sorted = PHONEME_LABELS_SORTED

        self.phoneme_to_id = {label: i for i,
                              label in enumerate(self.phonemes_sorted)}
        self.id_to_phoneme = self.phonemes_sorted
        self.labels_sorted = self.phonemes_sorted
        self.label_to_id = self.phoneme_to_id
        if (self.label_type == "voicing"):
            self.labels_sorted = ["uv", "v"]
            self.label_to_id = {"uv": 0, "v": 1}
        if (self.standardize and channel_means is None and channel_stds is None):
            self._calculate_standardization_params()
        elif (self.standardize and (channel_means is not None and channel_stds is not None)):
            self.channel_means = channel_means
            self.channel_stds = channel_stds
            self.broadcasted_stds = np.tile(
                self.channel_stds, (self.points_per_sample, 1)).T
            self.broadcasted_means = np.tile(
                self.channel_means, (self.points_per_sample, 1)).T

    def load_phonemes_from_tsv(self, subject, session, task, run):
        events_df = self._load_events(subject, session, task, run)
        events_df = events_df[events_df["kind"] == "phoneme"]
        events_df = events_df[events_df["segment"] != "oov_S"]
        events_df = events_df[events_df["segment"] != "sil"]
        phonemes = events_df["segment"].values
        onsets = events_df["timemeg"].values
        return phonemes, onsets

    def __getitem__(self, idx):
        # returns channels x time
        data, label, info = super().__getitem__(idx)

        phoneme_full = label
        phoneme = phoneme_full.split("_")[0]
        info["phoneme_full"] = phoneme_full

        phoneme_id = self.phoneme_to_id[phoneme]
        label_id = phoneme_id
        if (self.label_type == "voicing"):
            voicing_label = PHONATION_BY_PHONEME[phoneme]
            voicing_id = self.label_to_id[voicing_label]
            label_id = voicing_id

        if (self.include_info):
            return [data, torch.tensor(label_id), info]
        return [data, torch.tensor(label_id)]


if __name__ == "__main__":
    import time

    start_time = time.time()
    val_dataset = LibriBrainPhoneme(
        data_path="/Users/mirgan/LibriBrain/serialized/",
        partition="validation",
        preload_files=False,
    )
    test_dataset = LibriBrainPhoneme(
        data_path="/Users/mirgan/LibriBrain/serialized/",
        partition="test",
        preload_files=False,
    )
    print("len(val_dataset): ", len(val_dataset))
    print("len(test_dataset): ", len(test_dataset))

    label_counts = torch.zeros(len(val_dataset.labels_sorted))
    start_time = time.time()
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        label_counts[label] += 1
        if i % 1000 == 0:
            print(time.time() - start_time)
            start_time = time.time()
    print(label_counts)
