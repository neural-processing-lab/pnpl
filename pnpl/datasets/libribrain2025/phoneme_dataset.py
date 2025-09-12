import os
import warnings
from pnpl.datasets.libribrain2025.constants import PHONATION_BY_PHONEME
import numpy as np
import torch
from pnpl.datasets.libribrain2025.base import LibriBrainBase


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
        preload_files: bool = True,
        download: bool = True,
    ):
        """
        LibriBrain phoneme classification dataset.

        This dataset provides MEG data aligned to phoneme onsets for phoneme classification tasks.
        Each sample contains MEG data from tmin to tmax seconds relative to a phoneme onset.

        Args:
            data_path: Path where you wish to store the dataset. The local dataset structure 
                      will follow the same BIDS-like structure as the HuggingFace repo:
                      ```
                      data_path/
                      ├── {task}/                    # e.g., "Sherlock1"
                      │   └── derivatives/
                      │       ├── serialised/       # MEG data files
                      │       │   └── sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5
                      │       └── events/            # Event timing files  
                      │           └── sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv
                      ```
            partition: Convenient shortcut to specify train/validation/test split. Use "train", 
                      "validation", or "test". Instead of specifying run keys manually, you can use:
                      - partition="train": All runs except validation and test
                      - partition="validation": ('0', '11', 'Sherlock1', '2') 
                      - partition="test": ('0', '12', 'Sherlock1', '2')
            label_type: Type of labels to return. Options:
                       - "phoneme": Return phoneme labels (e.g., 'aa', 'ae', 'ah', etc.)
                       - "voicing": Return voicing labels derived from phonemes indicating voiced 
                         vs unvoiced phonemes. See https://en.wikipedia.org/wiki/Voice_(phonetics)
            preprocessing_str: By default, we expect files with preprocessing string 
                             "bads+headpos+sss+notch+bp+ds". This indicates the preprocessing steps:
                             bads+headpos+sss+notch+bp+ds means the data has been processed for 
                             bad channel removal, head position adjustment, signal-space separation, 
                             notch filtering, bandpass filtering, and downsampling.
            tmin: Start time of the sample in seconds relative to phoneme onset. For a phoneme 
                 at time T, you grab MEG data from T + tmin up to T + tmax.
            tmax: End time of the sample in seconds relative to phoneme onset. The number of 
                 timepoints per sample = int((tmax - tmin) * sfreq) where sfreq=250Hz.
            include_run_keys: List of specific sessions to include. Format per session: 
                            ('0', '1', 'Sherlock1', '1') = Subject 0, Session 1, Task Sherlock1, Run 1.
                            You can see all valid run keys by importing RUN_KEYS from 
                            pnpl.datasets.libribrain2025.constants.
            exclude_run_keys: List of sessions to exclude (same format as include_run_keys).
            exclude_tasks: List of task names to exclude (e.g., ['Sherlock1']).
            standardize: Whether to z-score normalize each channel's MEG data using mean and std 
                        computed across all included runs.
                        Formula: normalized_data[channel] = (raw_data[channel] - channel_mean[channel]) / channel_std[channel]
            clipping_boundary: If specified, clips all values to [-clipping_boundary, clipping_boundary]. 
                             This can help with outliers. Set to None for no clipping.
            channel_means: Pre-computed channel means for standardization. If provided along with 
                          channel_stds, these will be used instead of computing from the dataset.
            channel_stds: Pre-computed channel standard deviations for standardization.
            include_info: Whether to include additional info dict in each sample containing dataset name, 
                         subject, session, task, run, onset time, and full phoneme label (including 
                         word position indicators).
            preload_files: Whether to "eagerly" download all dataset files from HuggingFace when 
                          the dataset object is created (True) or "lazily" download files on demand (False). 
                          We recommend leaving this as True unless you have a specific reason not to.
            download: Whether to download files from HuggingFace if not found locally (True) or 
                     throw an error if files are missing locally (False).

        Returns:
            Data samples with shape (channels, time) where channels=306 MEG channels.
            Labels are integers corresponding to phoneme or voicing classes.
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
            preload_files=preload_files,
            download=download,
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

        self.phonemes_sorted = self._get_unique_phoneme_labels()
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

    def _get_unique_phoneme_labels(self):
        labels = set()
        for i in range(len(self)):
            labels.add(self.samples[i][5].split("_")[0])
        labels = list(labels)
        labels.sort()
        return labels

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
