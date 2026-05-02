"""
Schöffelen et al. (2019) "MOUS" MEG dataset.

Modern, mixin-based wrapper for the sentence-comprehension MEG dataset
released on the Radboud Data Repository
(https://webdav.data.ru.nl/dccn/DSC_3011020.09_236_v1/, ~200
participants split into ``A*`` listeners and ``V*`` readers, each
performing a task-specific block plus a shared resting-state block).

Layout differs slightly from LibriBrain / Armeni:

  - No session axis. The MOUS BIDS layout is
    ``sub-XXXX/meg/sub-XXXX_task-{auditory,visual,rest}_meg.ds``.
  - We synthesize a constant ``session = "01"`` so the
    ``(subject, session, task, run)`` 4-tuple convention from the rest
    of pnpl still applies.
  - Tasks are gated per subject (``A*`` ↔ ``auditory``, ``V*`` ↔
    ``visual``; ``rest`` for everyone).

Auth: see :class:`pnpl.datasets.mixins.RadboudDownloadMixin`.
"""

from __future__ import annotations

import os
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
    SUBJECTS,
    TASKS,
    is_task_for_subject,
)


class Schoffelen2019(
    RadboudDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """Schöffelen 2019 (MOUS) continuous MEG dataset (CTF).

    Args:
        data_path: Local data directory (BIDS root).
        task: TaskProtocol-shaped task object (see
            :mod:`pnpl.tasks.schoffelen2019`).
        preprocessing: Preprocessing string used in derivative filenames.
        preprocessing_config: Preprocessing-step overrides.
        include_subjects / exclude_subjects: BIDS subject ids without
            the ``sub-`` prefix (e.g., ``"A2002"``, ``"V1117"``).
        include_tasks / exclude_tasks: ``"auditory"``, ``"visual"``,
            ``"rest"``. Tasks are auto-skipped for subjects that didn't
            perform them (see :func:`is_task_for_subject`).
        include_run_keys / exclude_run_keys: 4-tuples
            ``(subject, "01", task, "01")``.
        standardize / clipping_boundary / channel_means / channel_stds:
            See :class:`pnpl.datasets.mixins.StandardizationMixin`.
        include_info: If True, ``__getitem__`` returns ``(x, y, info)``.
        create_h5_if_missing: If True, materialize H5 on demand.
        download: If True, fetch missing files from Radboud WebDAV.
        preload_h5: Read H5 into RAM on first access.
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

        # MOUS has no run/session axis — synthesize constants.
        self._session_id = "01"
        self._run_id = "01"

        intended_run_keys = self._resolve_run_keys(
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
        )
        if not intended_run_keys:
            raise ValueError(
                "No Schoffelen2019 run keys match the specified configuration"
            )
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
                    f"Could not resolve Schoffelen2019 data for {run_key}: {exc}. Skipping."
                )
        if skipped:
            warnings.warn(f"Skipped Schoffelen2019 run keys: {skipped}")
        if not self.run_keys:
            raise ValueError("No valid Schoffelen2019 run keys found")

        first_h5 = self.get_h5_path(*self.run_keys[0], preprocessing=self.preprocessing)
        self.sfreq = self.get_sfreq_from_h5(first_h5)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError(
                "No Schoffelen2019 samples found for the specified configuration"
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
        include_tasks,
        exclude_tasks,
    ) -> List[tuple]:
        include_run_keys = [tuple(map(str, rk)) for rk in (include_run_keys or [])]
        exclude_set = {tuple(map(str, rk)) for rk in (exclude_run_keys or [])}

        subjects = list(include_subjects) if include_subjects else SUBJECTS
        tasks = list(include_tasks) if include_tasks else TASKS

        if include_run_keys:
            base = list(include_run_keys)
        else:
            base = [
                (str(s), self._session_id, str(t), self._run_id)
                for s in subjects
                for t in tasks
                if is_task_for_subject(s, t)
            ]

        # Apply exclude filters
        excl_subjects = set(map(str, exclude_subjects or []))
        excl_tasks = set(map(str, exclude_tasks or []))
        out = []
        for subject, session, task, run in base:
            if (subject, session, task, run) in exclude_set:
                continue
            if subject in excl_subjects:
                continue
            if task in excl_tasks:
                continue
            # When run keys are user-specified, still respect the
            # auditory/visual/rest design constraint.
            if not is_task_for_subject(subject, task):
                continue
            out.append((subject, session, task, run))
        return out

    # ------------------------------------------------------------------
    # BIDS path overrides — no session, CTF .ds directories
    # ------------------------------------------------------------------

    def get_meg_dir(self, subject: str, session: str = None) -> str:  # noqa: ARG002
        return os.path.join(self.data_path, f"sub-{subject}", "meg")

    def get_bids_raw_path(
        self, subject: str, session: str, task: str, run: str
    ) -> str:  # noqa: ARG002
        name = f"sub-{subject}_task-{task}_meg.ds"
        return os.path.join(self.get_meg_dir(subject), name)

    def get_events_path(
        self, subject: str, session: str, task: str, run: str  # noqa: ARG002
    ) -> str:
        fname = f"sub-{subject}_task-{task}_events.tsv"
        path = os.path.join(self.get_meg_dir(subject), fname)
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
        fname = f"sub-{subject}_task-{task}"
        if proc is not None:
            fname += f"_proc-{proc}"
        fname += "_meg.h5"
        return os.path.join(
            self.data_path,
            "derivatives",
            "serialised",
            f"sub-{subject}",
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
            f"sub-{subject}_task-{task}_proc-{preprocessing}_meg.{extension}"
        )
        return os.path.join(
            self.data_path,
            "derivatives",
            "preproc",
            f"sub-{subject}",
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
        """Load the raw CTF recording, ensuring the ``.ds`` directory
        is fully downloaded first."""
        import mne

        ds_path = self.get_bids_raw_path(subject, session, task, run)
        if not (os.path.isdir(ds_path) and os.listdir(ds_path)):
            if not self.download:
                raise FileNotFoundError(f"Schoffelen2019 .ds directory not found: {ds_path}")
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
                f"Could not resolve/process Schoffelen2019 data for "
                f"sub-{subject}/task-{task}"
            ) from exc
        return h5_path

    def _serialise_fif_to_h5(self, fif_path: str, output_h5_path: str) -> None:
        import mne

        from ...preprocessing.serialization import fif_to_h5

        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        fif_to_h5(raw, output_h5_path)

    def _load_raw_lazy_meg_only(
        self, subject: str, session: str, task: str, run: str,
    ):
        """Open the recording without preloading, pick MEG only.

        Caller iterates time chunks via ``crop`` + ``load_data`` so
        peak memory stays bounded for multi-GB recordings.
        """
        raw = self.load_raw_bids(subject, session, task, run, preload=False)
        try:
            raw.pick(picks="meg", exclude=[])
        except Exception:
            pass
        return raw

    def _preprocess_raw_to_h5(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        output_h5_path: str,
        chunk_seconds: float = 120.0,
    ) -> None:
        import gc

        import mne
        import numpy as np

        from ...preprocessing import Pipeline
        from ...preprocessing.config import (
            load_json_config,
            resolve_preprocessing_config,
        )
        from ...preprocessing.serialization import fif_to_h5

        raw_lazy = self._load_raw_lazy_meg_only(subject, session, task, run)

        if self.preprocessing is None:
            raw_lazy.load_data(verbose=False)
            os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
            fif_to_h5(raw_lazy, output_h5_path)
            return

        step_names = self.preprocessing.split("+")
        json_config = load_json_config(self.data_path)
        resolved = resolve_preprocessing_config(
            step_names=step_names,
            json_config=json_config,
            dataset_config=self.preprocessing_config,
        )

        # Chunk-wise pipeline application — see Armeni2022 for the
        # rationale. Each chunk is preloaded, processed (notch + bp +
        # ds), and discarded except for the downsampled output.
        from ..armeni2022.dataset import _chunk_boundaries
        duration = float(raw_lazy.times[-1])
        boundaries = _chunk_boundaries(duration, chunk_seconds)

        from contextlib import redirect_stdout
        import io

        try:
            from tqdm.auto import tqdm  # type: ignore

            pbar = tqdm(
                total=len(boundaries),
                desc=f"Preprocessing {self.preprocessing}",
                unit="chunk",
                leave=True,
            )
        except Exception:
            pbar = None

        processed_chunks: list = []
        for start, end in boundaries:
            chunk = raw_lazy.copy().crop(tmin=start, tmax=end)
            chunk.load_data(verbose=False)
            pipeline = Pipeline.from_string(
                self.preprocessing, config=resolved.config
            )
            with redirect_stdout(io.StringIO()):
                chunk = pipeline.run(
                    chunk,
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    bids_root=self.data_path,
                    verbose=False,
                )
            if chunk._data is not None and chunk._data.dtype != np.float32:
                chunk._data = chunk._data.astype(np.float32, copy=False)
            processed_chunks.append(chunk)
            if pbar is not None:
                pbar.update(1)
            gc.collect()
        if pbar is not None:
            pbar.close()

        if len(processed_chunks) == 1:
            raw = processed_chunks[0]
        else:
            raw = mne.concatenate_raws(processed_chunks)
        del processed_chunks
        gc.collect()

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
                "dataset": "schoffelen2019",
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
            return 273
        return int(self.get_h5_dataset(self.run_keys[0]).shape[0])

    @property
    def n_times(self) -> int:
        return self.points_per_sample


__all__ = ["Schoffelen2019", "SUBJECTS", "TASKS"]
