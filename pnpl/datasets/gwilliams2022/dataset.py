"""
Gwilliams et al. 2022 / MEG-MASC dataset.

Modern, mixin-based wrapper for the MEG-MASC release on OSF
(https://osf.io/ag3kj/, https://arxiv.org/abs/2208.11488). Mirrors the
shape of ``LibriBrain`` and ``RIDDLE``: composes the standard pnpl
mixins, takes a ``TaskProtocol``-shaped ``task`` object, and supports
on-demand download via :class:`OSFDownloadMixin`.

MEG-MASC stores raw recordings as KIT/Yokogawa ``.con`` files (with
co-located ``markers.mrk`` and ``acq-ELP/HSP_headshape.pos`` files),
not Elekta ``.fif``, so this class overrides
:meth:`get_bids_raw_path` and :meth:`load_raw_bids` from
:class:`BIDSMixin` to use :func:`mne.io.read_raw_kit`.

Example:
    >>> from pnpl.datasets.gwilliams2022 import Gwilliams2022
    >>> from pnpl.tasks.gwilliams2022 import PhonemeClassification
    >>> ds = Gwilliams2022(
    ...     data_path="./meg_masc",
    ...     task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    ...     include_subjects=["01"],
    ...     include_sessions=["0"],
    ...     download=True,
    ... )
    >>> x, y = ds[0]
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
    OSFDownloadMixin,
    StandardizationMixin,
)
from .constants import (
    OSF_PROJECT_FALLBACKS,
    OSF_PROJECT_ID,
    SESSIONS,
    SUBJECTS,
    TASKS,
)


# Filename pattern for events.tsv files inside the BIDS layout. The four
# named groups become the run key.
_EVENTS_PATTERN = re.compile(
    r"^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)"
    r"_task-(?P<task>[^_]+)_events\.tsv$"
)


class Gwilliams2022(
    OSFDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """MEG-MASC continuous MEG dataset.

    Args:
        data_path: Local data directory. Files are arranged in BIDS layout
            here (matching what OSF provides). Created if missing.
        task: Object implementing :class:`pnpl.tasks.base.TaskProtocol`.
            See :mod:`pnpl.tasks.gwilliams2022` for ready-made tasks.
        preprocessing: Preprocessing string used in derivative filenames
            (e.g. ``"notch+bp+ds"``). When ``None``, the raw KIT recording
            is materialized to H5 unchanged.
        preprocessing_config: Optional preprocessing-step overrides
            forwarded to :class:`pnpl.preprocessing.Pipeline`.
        include_subjects / exclude_subjects: BIDS subject ids without the
            ``sub-`` prefix (e.g. ``"01"``).
        include_sessions / exclude_sessions: ``"0"`` or ``"1"``.
        include_tasks / exclude_tasks: Task ids in MEG-MASC's BIDS layout
            (``"0"``..``"3"`` — one per story).
        include_run_keys / exclude_run_keys: Tuples of
            ``(subject, session, task, run)`` to include/exclude. MEG-MASC
            uses a single run per task, so ``run`` is always ``"01"``.
        standardize / clipping_boundary / channel_means / channel_stds:
            See :class:`pnpl.datasets.mixins.StandardizationMixin`.
        include_info: If True, ``__getitem__`` returns ``(x, y, info)``.
        create_h5_if_missing: If True (default), materialize the cached H5
            from a local preprocessed FIF or — failing that — by running
            the preprocessing pipeline against the raw KIT recording.
        download: If True, fetch missing files from OSF on demand.
    """

    OSF_PROJECT_ID = OSF_PROJECT_ID
    OSF_PROJECT_FALLBACKS = OSF_PROJECT_FALLBACKS

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

        # MEG-MASC has no run dimension in its BIDS layout — derive a fixed
        # run id so the standard 4-tuple run key carries through unchanged.
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
            raise ValueError("No MEG-MASC run keys match the specified configuration")
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
                    f"Could not resolve MEG-MASC data for {run_key}: {exc}. Skipping."
                )

        if skipped:
            warnings.warn(f"Skipped MEG-MASC run keys: {skipped}")
        if not self.run_keys:
            raise ValueError("No valid MEG-MASC run keys found")

        first_h5 = self.get_h5_path(*self.run_keys[0], preprocessing=self.preprocessing)
        self.sfreq = self.get_sfreq_from_h5(first_h5)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError("No MEG-MASC samples found for the specified configuration")

        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )
        if standardize and channel_means is None and channel_stds is None:
            self._calculate_standardization_params()

    # ------------------------------------------------------------------
    # Run-key discovery and filtering
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
            # Fully specified — construct candidate run keys directly so we
            # can skip the multi-minute global OSF manifest walk. Files
            # that don't actually exist will be caught (and skipped with a
            # warning) by ``_ensure_h5_available``.
            base = [
                (str(s), str(ses), str(t), self._run_id)
                for s in include_subjects
                for ses in include_sessions
                for t in include_tasks
            ]
        else:
            base = self._discover_run_keys()

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

    def _discover_run_keys(self) -> List[tuple]:
        local = self._discover_local_run_keys()
        if not self.download:
            return local

        try:
            remote = self._discover_remote_run_keys()
        except Exception as exc:
            if local:
                warnings.warn(
                    "Failed to query OSF for MEG-MASC manifest; falling back to "
                    f"local discovery only ({exc})."
                )
                return local
            raise
        return sorted(set(local).union(remote))

    def _discover_local_run_keys(self) -> List[tuple]:
        run_keys: set[tuple] = set()
        if not os.path.isdir(self.data_path):
            return []
        # Only events.tsv files (one per task) are needed to enumerate runs.
        # Walk the BIDS layout: data_path/sub-*/ses-*/meg/*_events.tsv
        for sub_entry in os.scandir(self.data_path):
            if not (sub_entry.is_dir() and sub_entry.name.startswith("sub-")):
                continue
            for ses_entry in os.scandir(sub_entry.path):
                if not (ses_entry.is_dir() and ses_entry.name.startswith("ses-")):
                    continue
                meg_dir = os.path.join(ses_entry.path, "meg")
                if not os.path.isdir(meg_dir):
                    continue
                for fname in os.listdir(meg_dir):
                    rk = self._run_key_from_events_filename(fname)
                    if rk is not None:
                        run_keys.add(rk)
        return sorted(run_keys)

    def _discover_remote_run_keys(self) -> List[tuple]:
        run_keys: set[tuple] = set()
        for rel_path in self.list_remote_files():
            fname = rel_path.rsplit("/", 1)[-1]
            rk = self._run_key_from_events_filename(fname)
            if rk is not None:
                run_keys.add(rk)
        return sorted(run_keys)

    def _run_key_from_events_filename(self, fname: str) -> Optional[tuple]:
        match = _EVENTS_PATTERN.match(fname)
        if match is None:
            return None
        return (
            match.group("subject"),
            match.group("session"),
            match.group("task"),
            self._run_id,
        )

    # ------------------------------------------------------------------
    # BIDS path overrides — KIT files instead of FIF
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
    ) -> str:  # noqa: ARG002 — run unused (MASC has no run dimension)
        fname = f"sub-{subject}_ses-{session}_task-{task}_meg.con"
        return os.path.join(self.get_meg_dir(subject, session), fname)

    def get_markers_path(
        self, subject: str, session: str, task: str
    ) -> str:
        fname = f"sub-{subject}_ses-{session}_task-{task}_markers.mrk"
        return os.path.join(self.get_meg_dir(subject, session), fname)

    def get_elp_path(self, subject: str, session: str) -> str:
        fname = f"sub-{subject}_ses-{session}_acq-ELP_headshape.pos"
        return os.path.join(self.get_meg_dir(subject, session), fname)

    def get_hsp_path(self, subject: str, session: str) -> str:
        fname = f"sub-{subject}_ses-{session}_acq-HSP_headshape.pos"
        return os.path.join(self.get_meg_dir(subject, session), fname)

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
        run: str,  # noqa: ARG002 — single run per task in MASC
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
        run: str,  # noqa: ARG002
        preload: bool = True,
    ):
        """Load the raw KIT recording, fetching it (and the marker /
        head-shape sidecars KIT requires) from OSF if necessary."""
        import mne

        con_path = self.get_bids_raw_path(subject, session, task, run)
        mrk_path = self.get_markers_path(subject, session, task)
        elp_path = self.get_elp_path(subject, session)
        hsp_path = self.get_hsp_path(subject, session)

        for path in (con_path, mrk_path, elp_path, hsp_path):
            if not os.path.exists(path):
                if not self.download:
                    raise FileNotFoundError(f"MEG-MASC file not found: {path}")
                self.ensure_file(path)

        # MEG-MASC ships head-shape data with a ``.pos`` extension while MNE's
        # KIT loader only accepts ``.elp`` / ``.hsp`` / ``.mat`` / ``.txt``.
        # The contents are FastSCAN ASCII either way, so parse them into
        # ndarrays and hand those to ``read_raw_kit``.
        elp = _read_fastscan_points(elp_path)
        hsp = _read_fastscan_points(hsp_path)

        return mne.io.read_raw_kit(
            con_path,
            mrk=mrk_path,
            elp=elp,
            hsp=hsp,
            preload=preload,
            verbose=False,
        )

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

        # Try a matching preprocessed FIF first (cheaper than re-running the
        # pipeline from raw).
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

        # Fall back to running the pipeline against the raw KIT recording.
        try:
            self._preprocess_raw_to_h5(subject, session, task, run, h5_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Neither H5, matching preprocessed FIF, nor raw KIT recording could "
                f"be resolved/processed for sub-{subject}/ses-{session}/task-{task}"
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

            # Cache the preprocessed FIF alongside the raw layout so future
            # constructions skip the pipeline.
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
                "dataset": "gwilliams2022",
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
            return 208  # MEG-MASC uses a 208-channel KIT system
        return int(self.get_h5_dataset(self.run_keys[0]).shape[0])

    @property
    def n_times(self) -> int:
        return self.points_per_sample


def _read_fastscan_points(path: str) -> np.ndarray:
    """Parse a FastSCAN/Polhemus-style ASCII point file (``.pos`` /
    ``.elp`` / ``.hsp``). Comments start with ``%``; remaining lines hold
    one ``x y z`` triple per line in millimetres."""
    pts: list[list[float]] = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not pts:
        raise ValueError(f"No 3D points parsed from {path}")
    arr = np.asarray(pts, dtype=float)
    # FastSCAN files report points in millimetres; MNE expects metres.
    return arr / 1000.0


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


__all__ = [
    "Gwilliams2022",
    "SUBJECTS",
    "SESSIONS",
    "TASKS",
]
