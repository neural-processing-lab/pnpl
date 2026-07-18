"""
PNPL competition utilities.

Helpers for building and uploading Kaggle submissions for the PNPL 2026
competition. See :func:`write_submission` and :func:`submit_to_kaggle`.
"""

from .submission import (
    PNPL_2026_COMPETITIONS,
    PRIMARY_VOCAB,
    SECONDARY_VOCAB,
    SECONDARY_VOCAB_PREFIX,
    KaggleSubmissionResult,
    SubmissionError,
    load_vocabulary,
    resolve_competition,
    submit_to_kaggle,
    write_submission,
)
from .holdout import (
    BROAD_SUBJECTS,
    DEEP_SUBJECTS,
    HOLDOUT_REPO,
    N_SUBJECTS,
    SOURCES,
    TRACKS,
    WINDOW_SECONDS,
    HoldoutError,
    LibriBrainCompetitionHoldout,
)

__all__ = [
    # submission format + upload
    "PRIMARY_VOCAB",
    "SECONDARY_VOCAB",
    "SECONDARY_VOCAB_PREFIX",
    "SubmissionError",
    "KaggleSubmissionResult",
    "load_vocabulary",
    "submit_to_kaggle",
    "resolve_competition",
    "PNPL_2026_COMPETITIONS",
    "write_submission",
    # holdout data -> canonical rows
    "LibriBrainCompetitionHoldout",
    "HoldoutError",
    "TRACKS",
    "DEEP_SUBJECTS",
    "BROAD_SUBJECTS",
    "SOURCES",
    "N_SUBJECTS",
    "HOLDOUT_REPO",
    "WINDOW_SECONDS",
]
