"""
Pallier et al. (2025) "LittlePrince — Listen" MEG dataset.

Modern, mixin-based wrapper for OpenNeuro
`ds007523 <https://openneuro.org/datasets/ds007523/versions/1.0.1>`_
(``LittlePrince_MEG_French_Listen_Pallier2025``).

Layout: 58 subjects × 1 session (``01``) × 1 task (``listen``) × 9 runs
(``01``..``09``). Each run is a single ``.fif`` recording on an Elekta
Neuromag TRIUX system (306 channels, 1000 Hz). Events.tsv rows are
word onsets only — ``trial_type='Word'``, ``stimulus`` is the spoken
French token.

The dataset class composes the standard pnpl mixins:

  - :class:`OpenNeuroDownloadMixin` — fetches missing files from
    OpenNeuro's public S3 bucket on demand (no auth).
  - :class:`StandardizationMixin` — per-channel z-scoring + clipping.
  - :class:`ContinuousH5Mixin` — windowed reads from H5.
  - :class:`BIDSMixin` — BIDS path resolution.

Each subject ships ~8 GB of raw FIF (9 runs × ~0.8–1.0 GB). Full
dataset is ~478 GB, so start with a single subject / single run
before scaling up.

Example:
    >>> from pnpl.datasets import Pallier2025
    >>> from pnpl.tasks.pallier2025 import WordDetection
    >>> ds = Pallier2025(
    ...     data_path="./data/pallier2025",
    ...     task=WordDetection(tmin=0.0, tmax=3.0),
    ...     include_subjects=["01"],
    ...     include_runs=["01"],
    ... )
    >>> x, y = ds[0]
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..mixins import (
    BIDSMixin,
    ContinuousH5Mixin,
    OpenNeuroDownloadMixin,
    StandardizationMixin,
)
from .constants import (
    OPENNEURO_DATASET_ID,
    OPENNEURO_SNAPSHOT_TAG,
    RUNS,
    SESSIONS,
    SUBJECTS,
    TASKS,
)


class Pallier2025(
    OpenNeuroDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset,
):
    """LittlePrince audiobook-listening continuous MEG dataset.

    Args:
        data_path: Local data directory (BIDS root mirroring the
            OpenNeuro release). Created if missing.
        task: Object implementing :class:`pnpl.tasks.base.TaskProtocol`.
            See :mod:`pnpl.tasks.pallier2025` for ready-made tasks.
        preprocessing: Preprocessing string used in derivative
            filenames. Defaults to ``"notch+bp+ds"``. The companion
            paper (d'Ascoli et al., 2025, Nat Commun 16:10521) uses a
            0.1–40 Hz bandpass and 50 Hz resample with no notch / SSS,
            which can be reproduced via
            ``preprocessing="bp+ds"`` together with
            ``preprocessing_config={"bp": {"l_freq": 0.1, "h_freq": 40.0},
            "ds": {"sfreq": 50.0}}``.
        preprocessing_config: Optional preprocessing-step overrides
            forwarded to :class:`pnpl.preprocessing.Pipeline`.
        include_subjects / exclude_subjects: BIDS subject ids without
            the ``sub-`` prefix (``"01"``..``"58"``).
        include_sessions / exclude_sessions: ``"01"`` (the only one).
        include_tasks / exclude_tasks: ``"listen"`` (the only one).
        include_runs / exclude_runs: ``"01"``..``"09"``.
        include_run_keys / exclude_run_keys: 4-tuples
            ``(subject, session, task, run)`` for fully-specified
            inclusion/exclusion. Wins over the per-axis filters.
        standardize / clipping_boundary / channel_means / channel_stds:
            See :class:`pnpl.datasets.mixins.StandardizationMixin`.
        include_info: If True, ``__getitem__`` returns ``(x, y, info)``.
        create_h5_if_missing: If True (default), materialize the
            cached H5 from a local preprocessed FIF or — failing that
            — by running the preprocessing pipeline against the raw
            FIF.
        download: If True, fetch missing files from OpenNeuro on
            demand.
        preload_h5: Read each H5 into RAM on first access.
    """

    OPENNEURO_DATASET_ID = OPENNEURO_DATASET_ID
    OPENNEURO_SNAPSHOT_TAG = OPENNEURO_SNAPSHOT_TAG

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
        include_runs: Optional[Sequence[str]] = None,
        exclude_runs: Optional[Sequence[str]] = None,
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

        intended_run_keys = self._resolve_run_keys(
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
            include_sessions=include_sessions,
            exclude_sessions=exclude_sessions,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
            include_runs=include_runs,
            exclude_runs=exclude_runs,
        )
        if not intended_run_keys:
            raise ValueError("No Pallier2025 run keys match the specified configuration")
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
                    f"Could not resolve Pallier2025 data for {run_key}: {exc}. Skipping."
                )

        if skipped:
            warnings.warn(f"Skipped Pallier2025 run keys: {skipped}")
        if not self.run_keys:
            raise ValueError("No valid Pallier2025 run keys found")

        first_h5 = self.get_h5_path(*self.run_keys[0], preprocessing=self.preprocessing)
        self.sfreq = self.get_sfreq_from_h5(first_h5)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)

        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError(
                "No Pallier2025 samples found for the specified configuration"
            )

        # Drop samples whose window doesn't fully fit inside the recording.
        # The audiobook tasks use long windows (tmax=3.0 by default), so a
        # handful of late events would otherwise return short slices and
        # break batch collation.
        self.samples = self._filter_in_bounds_samples(self.samples)
        if not self.samples:
            raise ValueError(
                "All Pallier2025 samples were dropped as out-of-bounds — "
                "check tmin/tmax against the recording duration."
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
        include_runs,
        exclude_runs,
    ) -> List[tuple]:
        include_run_keys = [tuple(map(str, rk)) for rk in (include_run_keys or [])]
        exclude_set = {tuple(map(str, rk)) for rk in (exclude_run_keys or [])}

        if include_run_keys:
            base = list(include_run_keys)
        else:
            subjects = list(include_subjects) if include_subjects else SUBJECTS
            sessions = list(include_sessions) if include_sessions else SESSIONS
            tasks = list(include_tasks) if include_tasks else TASKS
            runs = list(include_runs) if include_runs else RUNS
            base = [
                (str(s), str(ses), str(t), str(r))
                for s in subjects
                for ses in sessions
                for t in tasks
                for r in runs
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
            include_runs=include_runs,
            exclude_runs=exclude_runs,
        )

    # ------------------------------------------------------------------
    # BIDS path overrides — events / cached H5 / preprocessed FIF
    # ------------------------------------------------------------------

    def get_meg_dir(self, subject: str, session: str) -> str:
        return os.path.join(
            self.data_path,
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
        )

    def get_events_path(
        self, subject: str, session: str, task: str, run: str,
    ) -> str:
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
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
        run: str,
        preprocessing: Optional[str] = None,
    ) -> str:
        proc = preprocessing if preprocessing is not None else getattr(self, "preprocessing", None)
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
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
        run: str,
        preprocessing: str,
        extension: str = "fif",
    ) -> str:
        fname = (
            f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
            f"_proc-{preprocessing}_meg.{extension}"
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

    # ------------------------------------------------------------------
    # Raw FIF loading
    # ------------------------------------------------------------------

    def load_raw_bids(
        self, subject: str, session: str, task: str, run: str,
        preload: bool = True,
    ):
        """Load the raw Elekta FIF, allowing the Internal Active Shielding
        flag MNE refuses by default.

        ds007523 was recorded on a Neuromag TRIUX with active shielding
        enabled, which sets a header tag MNE flags as needing MaxFilter.
        We don't run SSS in the default pipeline (the companion paper
        deliberately skips it), so we acknowledge the flag and pass the
        raw signal through to notch + bandpass + resample.
        """
        import mne
        import os

        raw_path = self.get_bids_raw_path(subject, session, task, run)
        if not os.path.exists(raw_path):
            if self.download:
                raw_path = self.ensure_file(raw_path)
            else:
                raise FileNotFoundError(f"Raw BIDS file not found: {raw_path}")
        return mne.io.read_raw_fif(
            raw_path, preload=preload, allow_maxshield=True, verbose=False,
        )

    # ------------------------------------------------------------------
    # H5 materialization
    # ------------------------------------------------------------------

    def _ensure_h5_available(
        self, subject: str, session: str, task: str, run: str,
    ) -> str:
        h5_path = self.get_h5_path(
            subject, session, task, run, preprocessing=self.preprocessing,
        )
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

        # Fall back to running the pipeline against the raw FIF.
        try:
            self._preprocess_raw_to_h5(subject, session, task, run, h5_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"Neither H5, matching preprocessed FIF, nor raw FIF could "
                f"be resolved/processed for sub-{subject}/ses-{session}/"
                f"task-{task}/run-{run}"
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
        """Open the raw FIF without preloading and pick MEG channels.

        EOG / ECG / STIM / misc channels are dropped so the chunk-wise
        preprocessing only filters MEG signals (and ``fif_to_h5`` ends
        up with a clean MEG-only matrix). Returned Raw is not preloaded
        — caller iterates time chunks via ``crop`` + ``load_data`` so
        peak memory stays bounded for the ~600 MB raw recordings.
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

        # Run the pipeline chunk-wise so peak memory stays bounded.
        # Each chunk emerges at the target sample rate; concatenating
        # processed chunks rebuilds the full timeline.
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

    def _filter_in_bounds_samples(self, samples: list[tuple]) -> list[tuple]:
        sfreq = self.sfreq
        pps = self.points_per_sample
        offset_samples = int(self.tmin * sfreq)

        run_lengths: dict[tuple, int] = {}
        for rk in self.run_keys:
            run_lengths[rk] = int(self.get_h5_dataset(rk).shape[1])

        kept: list[tuple] = []
        dropped = 0
        for sample in samples:
            rk = tuple(sample[:4])
            n_times = run_lengths.get(rk)
            if n_times is None:
                dropped += 1
                continue
            onset_samples = int(float(sample[4]) * sfreq) + offset_samples
            if onset_samples < 0 or onset_samples + pps > n_times:
                dropped += 1
                continue
            kept.append(sample)

        if dropped:
            warnings.warn(
                f"Pallier2025: dropped {dropped} samples whose window "
                f"({self.tmin:+g}..{self.tmax:+g} s) does not fit within "
                f"the recording — kept {len(kept)} of {len(samples)}."
            )
        return kept

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
                "dataset": "pallier2025",
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
            return 306  # Elekta Neuromag TRIUX: 102 mags + 204 grads
        return int(self.get_h5_dataset(self.run_keys[0]).shape[0])

    @property
    def n_times(self) -> int:
        return self.points_per_sample


def _chunk_boundaries(duration: float, chunk_seconds: float) -> List[tuple]:
    """Return ``[(t0, t1), ...]`` covering ``[0, duration]`` such that
    every interval is at least ``chunk_seconds * 0.5`` long. A short
    remainder gets folded into the previous interval — otherwise the
    last chunk can be too brief for mne's notch / bp filter design and
    triggers ``filter_length is longer than the signal`` distortion."""
    if duration <= 0:
        return []
    min_chunk = chunk_seconds * 0.5
    out: list[tuple] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        out.append((start, end))
        start = end
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < min_chunk:
        last_end = out[-1][1]
        out.pop()
        out[-1] = (out[-1][0], last_end)
    return out


def _apply_component_filters(
    run_keys: List[tuple],
    *,
    include_subjects=None,
    exclude_subjects=None,
    include_sessions=None,
    exclude_sessions=None,
    include_tasks=None,
    exclude_tasks=None,
    include_runs=None,
    exclude_runs=None,
) -> List[tuple]:
    include_subjects = set(map(str, include_subjects or []))
    exclude_subjects = set(map(str, exclude_subjects or []))
    include_sessions = set(map(str, include_sessions or []))
    exclude_sessions = set(map(str, exclude_sessions or []))
    include_tasks = set(map(str, include_tasks or []))
    exclude_tasks = set(map(str, exclude_tasks or []))
    include_runs = set(map(str, include_runs or []))
    exclude_runs = set(map(str, exclude_runs or []))

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
        if include_runs and run not in include_runs:
            continue
        if run in exclude_runs:
            continue
        out.append((subject, session, task, run))
    return out


__all__ = ["Pallier2025", "SUBJECTS", "SESSIONS", "TASKS", "RUNS"]
