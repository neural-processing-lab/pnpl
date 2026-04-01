import torch
import numpy as np
from pnpl.datasets.libribrain2025.base import LibriBrainBase


class LibriBrainSentence(LibriBrainBase):
    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
        preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
        # Note: tmin/tmax are usually fixed window sizes in the base class.
        # For sentences of varying length, we will handle slicing in __getitem__.
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
        # We initialize with tmin/tmax = 0 as placeholders;
        # sentence duration is dynamic.
        super().__init__(
            data_path=data_path,
            partition=partition,
            preprocessing_str=preprocessing_str,
            tmin=0.0,
            tmax=0.0,
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

        self.samples = []
        self.run_keys = []

        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                sentences_data = self.load_sentences_from_tsv(
                    subject, session, task, run)

                for entry in sentences_data:
                    # entry: (onset, duration, sentence_text)
                    sample = (subject, session, task, run,
                              entry['onset'], entry['duration'], entry['text'])
                    self.samples.append(sample)

                self.run_keys.append(run_key)
            except FileNotFoundError:
                continue

        if (self.standardize and channel_means is None):
            self._calculate_standardization_params()

    def load_sentences_from_tsv(self, subject, session, task, run):
        events_df = self._load_events(subject, session, task, run)

        # Filter for words only to construct the sentence string
        words_df = events_df[events_df["kind"] == "word"].copy()

        # Group by sentenceidx
        sentence_groups = words_df.groupby("sentenceidx")

        sentences_list = []
        for idx, group in sentence_groups:
            # Sort by timemeg to ensure correct word order
            group = group.sort_values("timemeg")

            # Concatenate words
            sentence_text = " ".join(group["segment"].astype(str).values)

            # Timing: Sentence starts at the first word onset
            # and ends after the last word's duration
            onset = group["timemeg"].iloc[0]
            offset = group["timemeg"].iloc[-1] + group["duration"].iloc[-1]
            duration = offset - onset

            sentences_list.append({
                "onset": onset,
                "duration": duration,
                "text": sentence_text
            })

        return sentences_list

    def __getitem__(self, idx):
        # Retrieve sample metadata
        subject, session, task, run, onset, duration, sentence_text = self.samples[idx]
        h5_dataset = self._get_open_h5_dataset((subject, session, task, run))

        # Calculate frame indices
        start_frame = int(onset * self.sfreq)
        # We use the sentence's specific duration rather than a fixed tmax
        end_frame = start_frame + int(duration * self.sfreq)

        data = h5_dataset[:, start_frame:end_frame]

        # Standardize if needed (note: broadcasted_means might need adjustment
        # for dynamic lengths, so we use the raw channel means/stds here)
        if self.standardize:
            # Reshape means for broadcasting: (Channels, 1)
            means = self.channel_means[:, np.newaxis]
            stds = self.channel_stds[:, np.newaxis]
            data = (data - means) / stds

        if self.clipping_boundary is not None:
            data = np.clip(data, -self.clipping_boundary,
                           self.clipping_boundary)

        output = {
            "meg": torch.tensor(data, dtype=torch.float32),
            "sentence": sentence_text
        }

        if self.include_info:
            output["info"] = {
                "subject": subject,
                "session": session,
                "task": task,
                "run": run,
                "onset": onset,
                "duration": duration
            }

        return output
