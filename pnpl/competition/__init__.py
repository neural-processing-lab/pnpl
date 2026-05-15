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

__all__ = [
    "PRIMARY_VOCAB",
    "SECONDARY_VOCAB",
    "SECONDARY_VOCAB_PREFIX",
    "SubmissionError",
    "load_vocabulary",
    "submit_to_kaggle",
    "write_submission",
]
