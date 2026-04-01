"""
Shared base class for LibriBrain dataset modules.

This keeps the dataset-specific wrappers aligned with the shared mixin and
download stack.
"""

import os

import pandas as pd
from torch.utils.data import Dataset

from ..mixins import BIDSMixin, ContinuousH5Mixin, HFDownloadMixin, StandardizationMixin
from ..utils import check_include_and_exclude_ids, include_exclude_ids
from .constants import RUN_KEYS, TEST_RUN_KEYS, VALIDATION_RUN_KEYS


class LibriBrainBase(
    HFDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """Base class used by the LibriBrain dataset implementations."""

    HUGGINGFACE_REPO = "pnpl/LibriBrain"
    HUGGINGFACE_FALLBACK_REPOS = ["pnpl/LibriBrain-Competition-2025"]

    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
        preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: list | None = None,
        exclude_run_keys: list | None = None,
        exclude_tasks: list | None = None,
        standardize: bool = True,
        clipping_boundary: float | None = 10.0,
        channel_means=None,
        channel_stds=None,
        include_info: bool = False,
        preload_files: bool = True,
        download: bool = True,
    ):
        self.data_path = data_path
        self.partition = partition
        self.preprocessing = preprocessing_str
        self.preprocessing_str = preprocessing_str
        self.tmin = tmin
        self.tmax = tmax
        self.include_info = include_info
        self.download = download
        self.standardize = standardize
        self.samples = []
        self.run_keys = []

        if self.download:
            os.makedirs(data_path, exist_ok=True)

        include_run_keys = include_run_keys or []
        exclude_run_keys = exclude_run_keys or []
        exclude_tasks = exclude_tasks or []

        if partition is not None:
            if include_run_keys or exclude_run_keys or exclude_tasks:
                raise ValueError(
                    "partition is a shortcut - include_run_keys, exclude_run_keys, "
                    "and exclude_tasks must be empty when partition is specified"
                )
            if partition == "train":
                exclude_run_keys = list(VALIDATION_RUN_KEYS) + list(TEST_RUN_KEYS)
            elif partition == "validation":
                include_run_keys = list(VALIDATION_RUN_KEYS)
            elif partition == "test":
                include_run_keys = list(TEST_RUN_KEYS)
            else:
                raise ValueError(
                    f"Invalid partition: {partition}. Must be train/validation/test"
                )

        include_run_keys = [tuple(run_key) for run_key in include_run_keys]
        exclude_run_keys = [tuple(run_key) for run_key in exclude_run_keys]
        check_include_and_exclude_ids(include_run_keys, exclude_run_keys, RUN_KEYS)

        intended_run_keys = include_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS
        )
        self.intended_run_keys = [
            run_key for run_key in intended_run_keys if run_key[2] not in exclude_tasks
        ]
        if not self.intended_run_keys:
            raise ValueError("No run keys match the specified configuration")

        self.init_continuous_h5()
        self.open_h5_datasets = {}

        if preload_files and download:
            self._prefetch_dataset_files()

        first_h5_path = None
        for run_key in self.intended_run_keys:
            candidate = self.get_h5_path(*run_key)
            try:
                if download:
                    candidate = self.ensure_file(candidate)
                elif not os.path.exists(candidate):
                    continue
                first_h5_path = candidate
                break
            except FileNotFoundError:
                continue

        if first_h5_path is None:
            raise FileNotFoundError(
                "Could not find any H5 files for the requested LibriBrain runs"
            )

        self.sfreq = self.get_sfreq_from_h5(first_h5_path)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )

    def _prefetch_dataset_files(self) -> None:
        file_paths = []
        for run_key in self.intended_run_keys:
            file_paths.append(self.get_h5_path(*run_key))
            file_paths.append(self.get_events_path(*run_key))
        self.prefetch_files(file_paths)

    def _calculate_standardization_params(self) -> None:
        self.calculate_standardization_params(self._get_open_h5_dataset)

    def _ids_to_h5_path(self, subject: str, session: str, task: str, run: str) -> str:
        return self.get_h5_path(subject, session, task, run)

    def _load_events(self, subject: str, session: str, task: str, run: str) -> pd.DataFrame:
        events_path = self.get_events_path(subject, session, task, run)
        if self.download:
            events_path = self.ensure_file(events_path)
        elif not os.path.exists(events_path):
            raise FileNotFoundError(events_path)
        return pd.read_csv(events_path, sep="\t")

    def _get_open_h5_dataset(self, run_key: tuple):
        h5_dataset = self.get_h5_dataset(run_key)
        self.open_h5_datasets[run_key] = h5_dataset
        return h5_dataset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.samples)}")

        subject, session, task, run, onset, label = self.samples[idx][:6]
        data = self.load_continuous_window(subject, session, task, run, onset)
        self._get_open_h5_dataset((subject, session, task, run))
        data = StandardizationMixin.standardize(self, data)

        info = {
            "dataset": "libribrain2025",
            "subject": subject,
            "session": session,
            "task": task,
            "run": run,
            "onset": onset,
        }
        return data, label, info
