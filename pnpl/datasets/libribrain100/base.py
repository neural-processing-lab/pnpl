"""
Base class for LibriBrain100 datasets.

Composes the same mixin stack used by ``LibriBrain`` plus
manifest-driven multi-repo download support: each manifest record
knows which Hugging Face repository owns its files, and the loader
fetches from that repo first, falling back to the other one
transparently.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..mixins import (
    BIDSMixin,
    ContinuousH5Mixin,
    HFDownloadMixin,
    StandardizationMixin,
)
from .constants import (
    DEFAULT_PREPROCESSING_STR,
    REPO_KEY_LIBRIBRAIN,
    REPO_KEY_LIBRIBRAIN2,
    REPO_KEY_TO_ID,
)
from .manifest import RunRecord, get_record, select_records
from .selectors import (
    CorpusArg,
    PartitionArg,
    SubjectsArg,
    normalize_corpus,
    normalize_partition,
    normalize_subjects,
    validate_selector_combination,
)


class LibriBrain100Base(
    HFDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """
    Common machinery for LibriBrain100 dataset variants.

    The class composes the standard pnpl mixins; individual subclasses
    only need to wire a task and implement ``__getitem__``.
    """

    # Primary + fallback repos exposed via HFDownloadMixin. They cover
    # the union of the two underlying repos so any file can be fetched
    # even if the per-record primary 404s. Per-file primary selection
    # happens in ``_schedule_download`` below.
    HUGGINGFACE_REPO = REPO_KEY_TO_ID[REPO_KEY_LIBRIBRAIN]
    HUGGINGFACE_FALLBACK_REPOS = [REPO_KEY_TO_ID[REPO_KEY_LIBRIBRAIN2]]

    def __init__(
        self,
        data_path: str,
        partition: PartitionArg = None,
        subjects: SubjectsArg = "all",
        corpus: CorpusArg = "all",
        preprocessing_str: Optional[str] = DEFAULT_PREPROCESSING_STR,
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: Optional[Sequence[Sequence[str]]] = None,
        exclude_run_keys: Optional[Sequence[Sequence[str]]] = None,
        exclude_tasks: Optional[Sequence[str]] = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = False,
        download: bool = True,
        preload_h5: bool = False,
    ):
        self.data_path = data_path
        self.partition = normalize_partition(partition)
        self.preprocessing = preprocessing_str
        self.preprocessing_str = preprocessing_str
        self.tmin = tmin
        self.tmax = tmax
        self.include_info = include_info
        self.download = download
        self.standardize_enabled_arg = standardize
        self.samples: list[tuple] = []
        self.run_keys: list[tuple[str, str, str, str]] = []

        if self.download:
            os.makedirs(data_path, exist_ok=True)

        partition_was_set = partition is not None
        include_run_keys = list(include_run_keys or [])
        exclude_run_keys = list(exclude_run_keys or [])
        exclude_tasks = list(exclude_tasks or [])
        if partition_was_set and (
            include_run_keys or exclude_run_keys or exclude_tasks
        ):
            raise ValueError(
                "partition is a shortcut — include_run_keys, "
                "exclude_run_keys, and exclude_tasks must be empty when "
                "partition is specified"
            )

        self._subjects_set = normalize_subjects(subjects)
        self._corpus_set = normalize_corpus(corpus)
        validate_selector_combination(
            subjects=self._subjects_set,
            corpus=self._corpus_set,
            partition=self.partition,
        )

        self._records = select_records(
            subjects=self._subjects_set,
            corpus=self._corpus_set,
            partition=self.partition,
            include_run_keys=include_run_keys or None,
            exclude_run_keys=exclude_run_keys or None,
            exclude_tasks=exclude_tasks or None,
        )
        if not self._records:
            raise ValueError(
                "No LibriBrain100 records match the requested configuration "
                f"(subjects={subjects!r}, corpus={corpus!r}, "
                f"partition={partition!r})."
            )

        self.intended_run_keys: list[tuple[str, str, str, str]] = [
            r.run_key for r in self._records
        ]

        # Map each absolute file path back to its manifest record so we
        # know which repo to try first when downloading. Populated lazily
        # as paths are produced; supports both H5 and events.tsv files.
        self._fpath_to_record: dict[str, RunRecord] = {}

        self.init_continuous_h5(preload_h5=preload_h5)
        self.open_h5_datasets: dict = {}

        if preload_files and download:
            self._prefetch_dataset_files()

        # Resolve sampling frequency from the first available H5; skip
        # records whose files cannot be downloaded so the dataset stays
        # usable while the LibriBrain2 upload is in flight.
        first_h5_path: Optional[str] = None
        resolved_records: list[RunRecord] = []
        skipped: list[tuple[str, str]] = []  # (run_key, reason)

        missing_excs = _missing_remote_exceptions()

        for record in self._records:
            h5_path = self._record_h5_path(record)
            try:
                if download:
                    h5_path = self.ensure_file(h5_path)
                elif not os.path.exists(h5_path):
                    raise FileNotFoundError(h5_path)
            except missing_excs as exc:
                skipped.append((record.run_key, str(exc)))
                continue
            resolved_records.append(record)
            if first_h5_path is None:
                first_h5_path = h5_path

        if skipped:
            warnings.warn(
                f"LibriBrain100: skipped {len(skipped)} run key(s) whose H5 "
                f"file could not be resolved (kept {len(resolved_records)} "
                f"of {len(self._records)}). First missing: {skipped[0][0]}"
            )
        if first_h5_path is None:
            raise FileNotFoundError(
                "Could not resolve any H5 files for the requested "
                "LibriBrain100 records. If the LibriBrain2 upload is "
                "still in progress, try again later or narrow the "
                "configuration to a corpus that is already available."
            )

        self._records = resolved_records
        self.run_keys = [r.run_key for r in resolved_records]

        self.sfreq = self.get_sfreq_from_h5(first_h5_path)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )

    # ------------------------------------------------------------------
    # Path helpers (override BIDSMixin / ContinuousH5Mixin defaults to
    # match the LibriBrain HF layout: per-corpus root, derivatives/
    # serialised + derivatives/events with the standard preprocessing
    # token)
    # ------------------------------------------------------------------

    def get_h5_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preprocessing: Optional[str] = None,
    ) -> str:
        proc = preprocessing if preprocessing is not None else self.preprocessing
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
        if proc is not None:
            fname += f"_proc-{proc}"
        fname += "_meg.h5"
        path = os.path.join(
            self.data_path, task, "derivatives", "serialised", fname,
        )
        try:
            self._fpath_to_record[path] = get_record((subject, session, task, run))
        except KeyError:
            pass
        return path

    def get_events_path(
        self, subject: str, session: str, task: str, run: str,
    ) -> str:
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        path = os.path.join(
            self.data_path, task, "derivatives", "events", fname,
        )
        try:
            self._fpath_to_record[path] = get_record((subject, session, task, run))
        except KeyError:
            pass
        return path

    def _record_h5_path(self, record: RunRecord) -> str:
        return self.get_h5_path(*record.run_key)

    def _record_events_path(self, record: RunRecord) -> str:
        return self.get_events_path(*record.run_key)

    # ------------------------------------------------------------------
    # Multi-repo download: pick the per-record primary so most fetches
    # hit on the first try; the inherited HFDownloadMixin retry chain
    # falls back to the other repo if needed.
    # ------------------------------------------------------------------

    def _schedule_download(self, fpath: str):
        rel_path = os.path.relpath(fpath, self.data_path).replace(os.path.sep, "/")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with self._lock:
            if fpath in self._download_futures:
                return self._download_futures[fpath]

            primary_repo, fallback_repos = self._repos_for_path(fpath)

            self._download_futures[fpath] = self._executor.submit(
                self._download_with_retry_static,
                fpath=fpath,
                rel_path=rel_path,
                data_path=self.data_path,
                primary_repo=primary_repo,
                fallback_repos=fallback_repos,
            )
            return self._download_futures[fpath]

    def _repos_for_path(self, fpath: str) -> tuple[str, list[str]]:
        """Pick the per-file primary HF repo, falling back to the other."""
        record = self._fpath_to_record.get(fpath)
        if record is not None:
            primary = REPO_KEY_TO_ID[record.repo]
            fallbacks = [
                REPO_KEY_TO_ID[k]
                for k in (REPO_KEY_LIBRIBRAIN, REPO_KEY_LIBRIBRAIN2)
                if REPO_KEY_TO_ID[k] != primary
            ]
            return primary, fallbacks
        # Unknown path (caller bypassed get_*_path) — fall back to the
        # class-level primary + fallback list.
        return self.HUGGINGFACE_REPO, list(self.HUGGINGFACE_FALLBACK_REPOS)

    def _prefetch_dataset_files(self) -> None:
        file_paths: list[str] = []
        for record in self._records:
            file_paths.append(self._record_h5_path(record))
            file_paths.append(self._record_events_path(record))
        self.prefetch_files(file_paths)

    # ------------------------------------------------------------------
    # Loaders shared with the existing LibriBrain wrappers
    # ------------------------------------------------------------------

    def _load_events(
        self, subject: str, session: str, task: str, run: str,
    ) -> pd.DataFrame:
        events_path = self.get_events_path(subject, session, task, run)
        if self.download:
            events_path = self.ensure_file(events_path)
        elif not os.path.exists(events_path):
            raise FileNotFoundError(events_path)
        return pd.read_csv(events_path, sep="\t")

    def _calculate_standardization_params(self) -> None:
        self.calculate_standardization_params(self._get_open_h5_dataset)

    def _get_open_h5_dataset(self, run_key: tuple):
        h5_dataset = self.get_h5_dataset(run_key)
        self.open_h5_datasets[run_key] = h5_dataset
        return h5_dataset

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> list[RunRecord]:
        """The manifest records actually loaded by this dataset."""
        return list(self._records)

    @property
    def n_channels(self) -> int:
        """Number of MEG channels (306 for MEGIN TRIUX Neo)."""
        return 306

    @property
    def n_times(self) -> int:
        return self.points_per_sample

    def __len__(self) -> int:
        return len(self.samples)


def _missing_remote_exceptions() -> tuple[type[BaseException], ...]:
    """Return the exception classes that mean "remote file not found".

    Includes ``FileNotFoundError`` for local-only paths, plus the
    relevant huggingface_hub error types so the loader can skip
    not-yet-uploaded records without crashing through the download
    stack. The HF imports are deferred so this module remains usable
    when ``huggingface_hub`` is not installed.
    """
    excs: list[type[BaseException]] = [FileNotFoundError]
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            RepositoryNotFoundError,
        )
    except Exception:  # pragma: no cover - depends on hf-hub install
        return tuple(excs)
    excs.append(EntryNotFoundError)
    excs.append(RepositoryNotFoundError)
    return tuple(excs)


__all__ = ["LibriBrain100Base"]
