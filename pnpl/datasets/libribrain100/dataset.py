"""
LibriBrain100 — task-based dataset entry point.

LibriBrain100 is the unified loader for the full LibriBrain release: a
virtual union of the original ``pnpl/LibriBrain`` Hugging Face dataset
and the extension repository ``pnpl/LibriBrain2``. It exposes the same
task-driven API as :class:`LibriBrain` plus two new selectors,
``subjects=`` and ``corpus=``, for picking subsets along the deep /
broad and per-corpus axes the dataset is built around.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .base import LibriBrain100Base
from .constants import DEFAULT_PREPROCESSING_STR
from .selectors import CorpusArg, PartitionArg, SubjectsArg


class LibriBrain100(LibriBrain100Base):
    """
    Task-driven dataset for the full LibriBrain100 release.

    Args:
        data_path: Local data directory. Files are arranged in a
            per-corpus BIDS-like layout that mirrors the Hugging Face
            tree (created lazily as files are downloaded).
        task: Object implementing :class:`pnpl.tasks.base.TaskProtocol`
            (e.g. :class:`pnpl.tasks.SpeechDetection`,
            :class:`pnpl.tasks.PhonemeClassification`,
            :class:`pnpl.tasks.WordDetection`).
        partition: ``"train"``, ``"validation"``, or ``"test"``. Aliases
            ``"val"``/``"valid"`` accepted. ``None`` means "no
            partition filter — apply only the explicit selectors".
        subjects: Subject selector. Accepts ``"all"`` (default),
            ``"deep"`` (sub-0, the deep single-subject component),
            ``"broad"`` (sub-1..32, the broad multi-subject
            component), an int, a string id (``"0"`` or ``"sub-0"``),
            or a list / range of ids.
        corpus: Corpus selector. Accepts ``"all"`` (default),
            ``"sherlock"``, ``"timit"``, ``"mocha"``,
            ``"podcasts"`` (aliases like ``"mocha-timit"``,
            ``"the_moth"`` accepted), or a list of those.
        preprocessing_str: Preprocessing token used in derivative
            filenames; defaults to ``"bads+headpos+sss+notch+bp+ds"``.
        include_run_keys / exclude_run_keys: 4-tuples
            ``(subject, session, task, run)`` for explicit
            inclusion/exclusion. Cannot be combined with ``partition``.
        exclude_tasks: Task tokens (e.g. ``["Sherlock1"]``) to drop.
        standardize / clipping_boundary / channel_means / channel_stds:
            See :class:`pnpl.datasets.mixins.StandardizationMixin`.
        include_info: If True, ``__getitem__`` returns ``(x, y, info)``.
        preload_files: Eagerly download every selected file at
            construction time (default ``False`` for LibriBrain100 —
            the dataset is large enough that lazy fetching is the
            usual choice).
        download: Enable downloading from Hugging Face.
        preload_h5: Read each H5 fully into RAM on first access.

    Notes:
        - The multi-subject (broad) data has no train partition by
          design; ``subjects="broad" + partition="train"`` raises
          :class:`ValueError`. For SFT workflows on broad subjects, use
          ``partition="validation"`` as your fine-tuning training set
          and ``partition="test"`` for evaluation.
        - Multi-subject data was only collected with the Sherlock
          stimuli; ``subjects="broad" + corpus="timit"`` (or any
          non-Sherlock corpus) raises :class:`ValueError`.

    Example:
        >>> from pnpl.datasets import LibriBrain100
        >>> from pnpl.tasks import SpeechDetection
        >>> ds = LibriBrain100(
        ...     data_path="./data/LibriBrain100",
        ...     task=SpeechDetection(tmin=0.0, tmax=0.5),
        ...     partition="train",
        ... )
        >>> x, y = ds[0]
    """

    def __init__(
        self,
        data_path: str,
        task,  # TaskProtocol
        partition: PartitionArg = None,
        subjects: SubjectsArg = "all",
        corpus: CorpusArg = "all",
        preprocessing_str: Optional[str] = DEFAULT_PREPROCESSING_STR,
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
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
        self.task = task
        self.preprocessing_config = preprocessing_config or {}

        # Defer tmin / tmax to the task (matches the LibriBrain pattern).
        tmin = float(getattr(task, "tmin", 0.0))
        tmax = float(getattr(task, "tmax", 0.5))

        super().__init__(
            data_path=data_path,
            partition=partition,
            subjects=subjects,
            corpus=corpus,
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
            preload_h5=preload_h5,
        )

        # Collect samples via the task; tasks read events.tsv files
        # through ``self.get_events_path`` and the inherited
        # ``ensure_file`` plumbing.
        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError(
                "No samples collected for the requested LibriBrain100 "
                "configuration."
            )

        if standardize and channel_means is None and channel_stds is None:
            self._calculate_standardization_params()

    # ------------------------------------------------------------------
    # Sample access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.samples):
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
            subject, session, task_name, run = sample[:4]
            onset = sample[4] if len(sample) > 4 else None
            info = {
                "dataset": "libribrain100",
                "subject": subject,
                "session": session,
                "task": task_name,
                "run": run,
                "onset": (
                    torch.tensor(float(onset), dtype=torch.float32)
                    if onset is not None
                    else None
                ),
            }
            return data, label, info
        return data, label

    @property
    def label_info(self):
        return self.task.label_info


__all__ = ["LibriBrain100"]
