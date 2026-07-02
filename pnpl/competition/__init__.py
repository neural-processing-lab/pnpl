"""
PNPL competition utilities.

Helpers for building and uploading Kaggle submissions for the PNPL 2026
competition. See :func:`write_submission` and :func:`submit_to_kaggle`.
"""

from .submission import (
    PRIMARY_VOCAB,
    SECONDARY_VOCAB,
    SECONDARY_VOCAB_PREFIX,
    SubmissionError,
    load_vocabulary,
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
    "load_vocabulary",
    "submit_to_kaggle",
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
