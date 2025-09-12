import os
import warnings
import numpy as np
import pandas as pd
import torch
from pnpl.datasets.libribrain2025.base import LibriBrainBase


class LibriBrainSpeech(LibriBrainBase):

    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
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
        oversample_silence_jitter: int = 0,
        preload_files: bool = True,
        stride=None,
        download: bool = True
    ):
        """
        LibriBrain speech vs silence classification dataset.

        This dataset provides MEG data segmented into time windows for binary classification of 
        speech vs silence. The dataset slides a time window across the continuous MEG data and 
        labels each window based on whether it contains predominantly speech or silence.

        Args:
            data_path: Path where you wish to store the dataset. The local dataset structure 
                      will follow the same BIDS-like structure as the HuggingFace repo:
                      ```
                      data_path/
                      ├── {task}/                    # e.g., "Sherlock1"
                      │   └── derivatives/
                      │       ├── serialised/       # MEG data files
                      │       │   └── sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5
                      │       └── events/            # Event files  
                      │           └── sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv
                      ```
            partition: Convenient shortcut to specify train/validation/test split. Use "train", 
                      "validation", or "test". Instead of specifying run keys manually, you can use:
                      - partition="train": All runs except validation and test
                      - partition="validation": ('0', '11', 'Sherlock1', '2') 
                      - partition="test": ('0', '12', 'Sherlock1', '2')
            preprocessing_str: By default, we expect files with preprocessing string 
                             "bads+headpos+sss+notch+bp+ds". This indicates the preprocessing steps:
                             bads+headpos+sss+notch+bp+ds means the data has been processed for 
                             bad channel removal, head position adjustment, signal-space separation, 
                             notch filtering, bandpass filtering, and downsampling.
            tmin: Start time of the sample in seconds relative to the sliding window start. 
                 Together with tmax, defines the time window size for each sample.
            tmax: End time of the sample in seconds relative to the sliding window start. 
                 The number of timepoints per sample = int((tmax - tmin) * sfreq) where sfreq=250Hz.
                 E.g., tmin=0, tmax=0.8 yields 200 timepoints per sample.
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
                         subject, session, task, run, and onset time of the sample.
            oversample_silence_jitter: Since the dataset is quite unbalanced (more speech than silence), 
                                     you may wish to oversample the silent portions during training. This 
                                     parameter allows you to specify a different stride for silent portions only, 
                                     effectively oversampling them. Set to 0 for no oversampling.
            preload_files: Whether to "eagerly" download all dataset files from HuggingFace when 
                          the dataset object is created (True) or "lazily" download files on demand (False). 
                          We recommend leaving this as True unless you have a specific reason not to.
            stride: Controls how far (in time) you move the sliding window between consecutive samples. 
                   Instead of jumping exactly one full time_window_samples worth (tmax-tmin; the default) 
                   each time, you can specify a smaller stride to get overlapping windows. If None, 
                   defaults to time_window_samples (no overlap).
            download: Whether to download files from HuggingFace if not found locally (True) or 
                     throw an error if files are missing locally (False).

        Returns:
            Data samples with shape (channels, time) where channels=306 MEG channels.
            Labels are arrays indicating speech (1) vs silence (0) for each timepoint in the sample.
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
            download=download
        )

        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")
        self.oversample_silence_jitter = oversample_silence_jitter

        self.stride = stride
        self.samples = []
        run_keys_missing = []
        self.run_keys = []
        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                self.speech_labels = self.get_speech_silence_labels_for_session(
                    subject, session, task, run)
                if self.oversample_silence_jitter > 0:
                    self._collect_speech_over_samples(
                        subject, session, task, run, self.speech_labels, self.oversample_silence_jitter)
                else:
                    self._collect_speech_samples(
                        subject, session, task, run, self.speech_labels, stride=self.stride)
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

        if (self.standardize and channel_means is None and channel_stds is None):
            self._calculate_standardization_params()
        elif (self.standardize and (channel_means is not None and channel_stds is not None)):
            self.channel_means = channel_means
            self.channel_stds = channel_stds
            self.broadcasted_stds = np.tile(
                self.channel_stds, (self.points_per_sample, 1)).T
            self.broadcasted_means = np.tile(
                self.channel_means, (self.points_per_sample, 1)).T

    def get_speech_silence_labels_for_session(self, subject, session, task, run):
        df = self._load_events(subject, session, task, run)

        # Convert times to samples, handling errors
        df['timemeg_samples'] = (pd.to_numeric(
            df['timemeg'], errors='coerce') * self.sfreq).astype(int)
        df['duration_samples'] = (pd.to_numeric(
            df['duration'], errors='coerce') * self.sfreq).astype(int)

        # Filter for silence and word entries
        silence_df = df[df['kind'] == 'silence']
        words_df = df[df['kind'] == 'word']

        if silence_df.empty or silence_df['timemeg_samples'].isnull().all() or silence_df[
            'duration_samples'].isnull().all():
            print("Warning: No valid silence entries found. Returning None.")
            return None

        if words_df.empty or words_df['timemeg_samples'].isnull().all() or words_df['duration_samples'].isnull().all():
            print("Warning: No valid word entries found. Returning None.")
            return None

        words_df = df[df['kind'] == 'word']

        max_word_sample_time = (words_df['timemeg_samples'] +
                                words_df['duration_samples']).max()

        max_silence_sample_time = (silence_df['timemeg_samples'] +
                                   silence_df['duration_samples']).max()

        min_word_sample_time = (words_df['timemeg_samples']).min()

        min_silence_sample_time = (silence_df['timemeg_samples']).min()

        # Create the array, initialize with 1s (assuming everything is speech initially)
        speech_labels = np.ones(max(max_word_sample_time, max_silence_sample_time) + 1, dtype=int)
        # Determine the minimum annotated sample time
        min_sample_time = min(min_word_sample_time, min_silence_sample_time)
        # Set to silence before we get any label
        speech_labels[0:min_sample_time] = 0
        # Fill in 0s for silence spans (adjusted for the offset)
        for index, row in silence_df.iterrows():
            start_sample = row['timemeg_samples']
            duration_samples = row['duration_samples']
            if not np.isnan(start_sample) and not np.isnan(duration_samples):
                end_sample = start_sample + duration_samples
                speech_labels[start_sample:end_sample] = 0

        return speech_labels

    def _collect_speech_samples(self, subject, session, task, run, speech_labels, stride = None):
        # Calculate the number of samples in the time window
        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)

        if stride is None:
            stride = time_window_samples

        for i in range(0, len(speech_labels), stride):
            sample_labels = speech_labels[i:i+time_window_samples]
            if len(sample_labels) < time_window_samples:
                continue
            self.samples.append(
                (subject, session, task, run, i / self.sfreq, sample_labels))

    def _collect_speech_over_samples(self, subject, session, task, run, speech_labels, silence_jitter=7, over_sample_category=1):
        # Calculate the number of samples in the time window
        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)

        # first collect the normal samples
        for i in range(0, len(speech_labels), time_window_samples):
            sample_labels = speech_labels[i:i+time_window_samples]
            if len(sample_labels) < time_window_samples:
                continue
            self.samples.append(
                (subject, session, task, run, i / self.sfreq, sample_labels))

        # now collect the over samples
        samples_step_size = time_window_samples
        i = 0
        self.segments_with_speech_counter = 0
        jitter_around_silence = False
        # Make sure to jitter around silence whenever a silence is found in the samples iteration
        while i < len(speech_labels):
            speech_label_segment = speech_labels[i:i + time_window_samples]
            # found rare silence, iterate sample at a time to oversample silence
            if speech_label_segment.sum() < time_window_samples and jitter_around_silence == False:
                jitter_around_silence = True
                first_zero_index = np.argmax(speech_label_segment == 0)
                i = i - ((time_window_samples - first_zero_index) - 1)
                samples_step_size = silence_jitter
            # back to no silence, so let's go back to 200 sampling rate step size
            if speech_label_segment.sum() == time_window_samples and jitter_around_silence == True:
                samples_step_size = time_window_samples
                jitter_around_silence = False

            sample_labels = speech_labels[i:i+time_window_samples]
            if len(sample_labels) < time_window_samples:
                break
            i += samples_step_size

            if over_sample_category == 1:
                if 0.3 < sample_labels.sum() / sample_labels.shape[0] < 0.5:
                    self.samples.append(
                        (subject, session, task, run, i / self.sfreq, sample_labels))
                if sample_labels.sum() == 0:
                    self.samples.append(
                        (subject, session, task, run, i / self.sfreq, sample_labels))

    def __getitem__(self, idx):
        # returns channels x time
        data, label, info = super().__getitem__(idx)
        if self.include_info:
            return [data, torch.tensor(label), info]
        return [data, torch.tensor(label)]


if __name__ == "__main__":
    import time

    start_time = time.time()
    dataset = LibriBrainSpeech(
        data_path="/Users/gilad/Desktop/Projects/PNPL/LibriBrain/serialized",
        preprocessing_str="bads+headpos+sss+notch+bp+ds",
        exclude_run_keys=[['0', '11', 'Sherlock1', '2'],
                          ['0', '12', 'Sherlock1', '2']],
        include_run_keys=[['0', '1', 'Sherlock1', '1'], ['0', '2', 'Sherlock1', '1'], ['0', '3', 'Sherlock1', '1'],
                          ['0', '4', 'Sherlock1', '1'], ['0', '5', 'Sherlock1', '1'], [
                              '0', '6', 'Sherlock1', '1'],
                          ['0', '7', 'Sherlock1', '1'], ['0', '8', 'Sherlock1', '1'], [
                              '0', '9', 'Sherlock1', '1'],
                          ['0', '10', 'Sherlock1', '1']],
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True)
    batch = next(iter(loader))
    label_counts = torch.zeros(2)
    start_time = time.time()
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_counts[label] += 1
        if i % 1000 == 0:
            print(time.time() - start_time)
            start_time = time.time()
    print(label_counts)
