"""
Armeni et al. (2022) MEG dataset.

Modern, mixin-based wrapper for the audiobook-listening dataset
released on the Radboud Data Repository
(https://webdav.data.ru.nl/dccn/DSC_3011085.05_995_v1/, 3 subjects, 10
sessions × ``compr`` task each, ~10 hours total per subject).

The Radboud release stores raw recordings as CTF ``.ds`` *directories*
(not single files), so this class overrides
:meth:`get_bids_raw_path` and :meth:`load_raw_bids` to use
:func:`mne.io.read_raw_ctf` and :meth:`RadboudDownloadMixin.ensure_directory`.

Auth (required when ``download=True``): set the env vars
``RADBOUD_USERNAME`` and ``RADBOUD_PASSWORD`` to your Radboud Data
Repository credentials. The Radboud DSC is *not* open access; you
need an approved data-sharing agreement.

Example:
    >>> import os
    >>> os.environ["RADBOUD_USERNAME"] = "you@orcid.org"
    >>> os.environ["RADBOUD_PASSWORD"] = "..."
    >>> from pnpl.datasets.armeni2022 import Armeni2022
    >>> from pnpl.tasks.armeni2022 import PhonemeClassification
    >>> ds = Armeni2022(
    ...     data_path="./data/armeni",
    ...     task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    ...     include_subjects=["001"],
    ...     include_sessions=["001"],
    ... )
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..mixins import (
    BIDSMixin,
    ContinuousH5Mixin,
    RadboudDownloadMixin,
    StandardizationMixin,
)
from .constants import (
    RADBOUD_DATASET_URL,
    SESSIONS,
    SUBJECTS,
    TASKS,
)


_EVENTS_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)"
    r"_task-(?P<task>[^_]+)_events\.tsv$"
)


class Armeni2022(
    RadboudDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """Armeni 2022 continuous MEG dataset (CTF, audiobook listening).

    Args:
        data_path: Local data directory (BIDS root mirroring the
            Radboud release). Created if missing.
        task: Object implementing :class:`pnpl.tasks.base.TaskProtocol`.
            See :mod:`pnpl.tasks.armeni2022` for ready-made tasks.
        preprocessing: Preprocessing string used in derivative filenames.
            ``None`` materializes H5 from the raw CTF unchanged.
        preprocessing_config: Optional preprocessing-step overrides.
        include_subjects / exclude_subjects: BIDS subject ids without
            the ``sub-`` prefix (``"001"``..``"003"``).
        include_sessions / exclude_sessions: ``"001"`` .. ``"010"``.
        include_tasks / exclude_tasks: Currently only ``"compr"``.
        include_run_keys / exclude_run_keys: 4-tuples
            ``(subject, session, task, run)``. Run is always ``"01"``
            (Armeni has no run dimension).
        standardize / clipping_boundary / channel_means / channel_stds:
            See :class:`pnpl.datasets.mixins.StandardizationMixin`.
        include_info: If True, ``__getitem__`` returns ``(x, y, info)``.
        create_h5_if_missing: If True (default), materialize the cached
            H5 from a local preprocessed FIF or, failing that, by
            running the preprocessing pipeline against the raw CTF.
        download: If True, fetch missing files from Radboud WebDAV.
        preload_h5: Read each H5 into RAM on first access.
    """

    RADBOUD_DATASET_URL = RADBOUD_DATASET_URL

    def __init__(
        self,
        data_path: str,
        task,  # TaskProtocol
        preprocessing: Optional[str] = "notch+bp+ds",
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
        include_subjects: Optional[Sequence[str]] = None,
        exclude_subjects: Optional[Sequence[str]] = None,
        include_sessions: Optional[Sequence[str]] = None,
        exclude_sessions: Optional[Sequence[str]] = None,
        include_tasks: Optional[Sequence[str]] = None,
        exclude_tasks: Optional[Sequence[str]] = None,
        include_run_keys: Optional[Sequence[tuple]] = None,
        exclude_run_keys: Optional[Sequence[tuple]] = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        create_h5_if_missing: bool = True,
        download: bool = True,
        preload_h5: bool = False,
    ):
        os.makedirs(data_path, exist_ok=True)

        self.data_path = data_path
        self.task = task
        self.preprocessing = preprocessing
        self.preprocessing_config = preprocessing_config or {}
        self.include_info = include_info
        self.create_h5_if_missing = create_h5_if_missing
        self.download = download

        self.tmin = float(getattr(task, "tmin", 0.0))
        self.tmax = float(getattr(task, "tmax", 0.5))
        self._run_id = "01"

        intended_run_keys = self._resolve_run_keys(
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
            include_sessions=include_sessions,
            exclude_sessions=exclude_sessions,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
        )
        if not intended_run_keys:
            raise ValueError("No Armeni2022 run keys match the specified configuration")
        self.intended_run_keys = intended_run_keys

        self.init_continuous_h5(preload_h5=preload_h5)

        self.run_keys: list[tuple] = []
        skipped: list[tuple] = []
        for run_key in self.intended_run_keys:
            try:
                self._ensure_h5_available(*run_key)
                self.run_keys.append(run_key)
            except FileNotFoundError as exc:
                skipped.append(run_key)
                warnings.warn(
                    f"Could not resolve Armeni2022 data for {run_key}: {exc}. Skipping."
                )
        if skipped:
            warnings.warn(f"Skipped Armeni2022 run keys: {skipped}")
        if not self.run_keys:
            raise ValueError("No valid Armeni2022 run keys found")

        first_h5 = self.get_h5_path(*self.run_keys[0], preprocessing=self.preprocessing)
        self.sfreq = self.get_sfreq_from_h5(first_h5)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError(
                "No Armeni2022 samples found for the specified configuration"
            )

        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )
        if standardize and channel_means is None and channel_stds is None:
            self._calculate_standardization_params()

    # ------------------------------------------------------------------
    # Run-key discovery
    # ------------------------------------------------------------------

    def _resolve_run_keys(
        self,
        *,
        include_run_keys,
        exclude_run_keys,
        include_subjects,
        exclude_subjects,
        include_sessions,
        exclude_sessions,
        include_tasks,
        exclude_tasks,
    ) -> List[tuple]:
        include_run_keys = [tuple(map(str, rk)) for rk in (include_run_keys or [])]
        exclude_set = {tuple(map(str, rk)) for rk in (exclude_run_keys or [])}

        if include_run_keys:
            base = list(include_run_keys)
        elif include_subjects and include_sessions and include_tasks:
            base = [
                (str(s), str(ses), str(t), self._run_id)
                for s in include_subjects
                for ses in include_sessions
                for t in include_tasks
            ]
        else:
            # Fall back to the full BIDS axis declared by the dataset
            # constants, then narrow with explicit filters. Avoids any
            # remote walk for the compact 3 × 10 × 1 = 30-run case.
            subjects = list(include_subjects) if include_subjects else SUBJECTS
            sessions = list(include_sessions) if include_sessions else SESSIONS
            tasks = list(include_tasks) if include_tasks else TASKS
            base = [
                (str(s), str(ses), str(t), self._run_id)
                for s in subjects
                for ses in sessions
                for t in tasks
            ]

        base = [rk for rk in base if rk not in exclude_set]
        return _apply_component_filters(
            base,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
            include_sessions=include_sessions,
            exclude_sessions=exclude_sessions,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
        )

    # ------------------------------------------------------------------
    # BIDS path overrides — CTF .ds directories instead of FIF files
    # ------------------------------------------------------------------

    def get_meg_dir(self, subject: str, session: str) -> str:
        return os.path.join(
            self.data_path,
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
        )

    def get_bids_raw_path(
        self, subject: str, session: str, task: str, run: str
    ) -> str:  # noqa: ARG002
        name = f"sub-{subject}_ses-{session}_task-{task}_meg.ds"
        return os.path.join(self.get_meg_dir(subject, session), name)

    def get_events_path(
        self, subject: str, session: str, task: str, run: str  # noqa: ARG002
    ) -> str:
        fname = f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
        path = os.path.join(self.get_meg_dir(subject, session), fname)
        if os.path.exists(path):
            return path
        if self.download:
            return self.ensure_file(path)
        return path

    def get_h5_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,  # noqa: ARG002
        preprocessing: Optional[str] = None,
    ) -> str:
        proc = preprocessing if preprocessing is not None else getattr(self, "preprocessing", None)
        fname = f"sub-{subject}_ses-{session}_task-{task}"
        if proc is not None:
            fname += f"_proc-{proc}"
        fname += "_meg.h5"
        return os.path.join(
            self.data_path,
            "derivatives",
            "serialised",
            f"sub-{subject}",
            f"ses-{session}",
            fname,
        )

    def get_preprocessed_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,  # noqa: ARG002
        preprocessing: str,
        extension: str = "fif",
    ) -> str:
        fname = (
            f"sub-{subject}_ses-{session}_task-{task}_proc-{preprocessing}_meg.{extension}"
        )
        return os.path.join(
            self.data_path,
            "derivatives",
            "preproc",
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
            fname,
        )

    def load_raw_bids(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preload: bool = True,
    ):
        """Load the raw CTF recording. The ``.ds`` "file" is actually a
        directory of binary chunks, so we ensure every contained file is
        downloaded first (a single ``ensure_directory`` call)."""
        import mne

        ds_path = self.get_bids_raw_path(subject, session, task, run)
        if not (os.path.isdir(ds_path) and os.listdir(ds_path)):
            if not self.download:
                raise FileNotFoundError(f"Armeni2022 .ds directory not found: {ds_path}")
            self.ensure_directory(ds_path)

        return mne.io.read_raw_ctf(ds_path, preload=preload, verbose=False)

    # ------------------------------------------------------------------
    # H5 materialization
    # ------------------------------------------------------------------

    def _ensure_h5_available(
        self, subject: str, session: str, task: str, run: str
    ) -> str:
        h5_path = self.get_h5_path(subject, session, task, run, preprocessing=self.preprocessing)
        if os.path.exists(h5_path):
            return h5_path

        last_error: Optional[Exception] = None
        if self.download:
            try:
                return self.ensure_file(h5_path)
            except Exception as exc:
                last_error = exc

        if not self.create_h5_if_missing:
            raise FileNotFoundError(h5_path) from last_error

        if self.preprocessing is not None:
            fif_path = self.get_preprocessed_path(
                subject, session, task, run,
                preprocessing=self.preprocessing,
                extension="fif",
            )
            if not os.path.exists(fif_path) and self.download:
                try:
                    fif_path = self.ensure_file(fif_path)
                except Exception as exc:
                    last_error = exc
            if os.path.exists(fif_path):
                self._serialise_fif_to_h5(fif_path, h5_path)
                return h5_path

        try:
            self._preprocess_raw_to_h5(subject, session, task, run, h5_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not resolve/process Armeni2022 data for "
                f"sub-{subject}/ses-{session}/task-{task}"
            ) from exc
        return h5_path

    def _serialise_fif_to_h5(self, fif_path: str, output_h5_path: str) -> None:
        import mne

        from ...preprocessing.serialization import fif_to_h5

        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        fif_to_h5(raw, output_h5_path)

    def _preprocess_raw_to_h5(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        output_h5_path: str,
    ) -> None:
        from ...preprocessing import Pipeline
        from ...preprocessing.config import (
            load_json_config,
            resolve_preprocessing_config,
        )
        from ...preprocessing.serialization import fif_to_h5

        raw = self.load_raw_bids(subject, session, task, run, preload=True)

        if self.preprocessing is not None:
            step_names = self.preprocessing.split("+")
            json_config = load_json_config(self.data_path)
            resolved = resolve_preprocessing_config(
                step_names=step_names,
                json_config=json_config,
                dataset_config=self.preprocessing_config,
            )
            pipeline = Pipeline.from_string(self.preprocessing, config=resolved.config)
            raw = pipeline.run(
                raw,
                subject=subject,
                session=session,
                task=task,
                run=run,
                bids_root=self.data_path,
                verbose=False,
            )

            fif_path = self.get_preprocessed_path(
                subject, session, task, run,
                preprocessing=self.preprocessing,
                extension="fif",
            )
            os.makedirs(os.path.dirname(fif_path), exist_ok=True)
            raw.save(fif_path, overwrite=True, verbose=False)

        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        fif_to_h5(raw, output_h5_path)

    def _calculate_standardization_params(self) -> None:
        def h5_loader(run_key):
            return self.get_h5_dataset(run_key)

        self.calculate_standardization_params(h5_loader)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self.samples)}"
            )
        sample = self.samples[idx]
        data = self.load_continuous_window_from_sample(sample)
        data = self.standardize(data)
        label = self.task.get_label(sample)

        data = torch.tensor(np.asarray(data), dtype=torch.float32)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        elif isinstance(label, (int, np.integer)):
            label = torch.tensor(int(label))

        if self.include_info:
            info = {
                "dataset": "armeni2022",
                "subject": sample[0],
                "session": sample[1],
                "task": sample[2],
                "run": sample[3],
                "onset": torch.tensor(sample[4], dtype=torch.float32),
                "label_value": sample[5],
            }
            return data, label, info
        return data, label

    @property
    def label_info(self):
        return self.task.label_info

    @property
    def n_channels(self) -> int:
        if not self.run_keys:
            return 273  # CTF275 axial gradiometer system; ~270 working channels
        return int(self.get_h5_dataset(self.run_keys[0]).shape[0])

    @property
    def n_times(self) -> int:
        return self.points_per_sample


def _apply_component_filters(
    run_keys: List[tuple],
    *,
    include_subjects=None,
    exclude_subjects=None,
    include_sessions=None,
    exclude_sessions=None,
    include_tasks=None,
    exclude_tasks=None,
) -> List[tuple]:
    include_subjects = set(map(str, include_subjects or []))
    exclude_subjects = set(map(str, exclude_subjects or []))
    include_sessions = set(map(str, include_sessions or []))
    exclude_sessions = set(map(str, exclude_sessions or []))
    include_tasks = set(map(str, include_tasks or []))
    exclude_tasks = set(map(str, exclude_tasks or []))

    out = []
    for subject, session, task, run in run_keys:
        if include_subjects and subject not in include_subjects:
            continue
        if subject in exclude_subjects:
            continue
        if include_sessions and session not in include_sessions:
            continue
        if session in exclude_sessions:
            continue
        if include_tasks and task not in include_tasks:
            continue
        if task in exclude_tasks:
            continue
        out.append((subject, session, task, run))
    return out


__all__ = ["Armeni2022", "SUBJECTS", "SESSIONS", "TASKS"]
