"""
Constants for the LibriBrain100 dataset.

LibriBrain100 is a virtual union of two Hugging Face dataset repos:

  - ``pnpl/LibriBrain``  — the original release (sub-0 × Sherlock1..7)
  - ``pnpl/LibriBrain2`` — the extension release (sub-0 ×
    Sherlock8/Sherlock9/TIMIT/MOCHATIMIT/Podcasts, plus sub-1..32 ×
    Sherlock1 ses-11/ses-12)

The folder/task token in HF for "MOCHA-TIMIT" is the unhyphenated
``MOCHATIMIT``; we accept both spellings as user-facing aliases but
the task token in run keys is always ``MOCHATIMIT``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hugging Face repo identifiers
# ---------------------------------------------------------------------------

REPO_LIBRIBRAIN = "pnpl/LibriBrain"
REPO_LIBRIBRAIN2 = "pnpl/LibriBrain2"

# Unique label per repo used internally on RunRecord.repo.
REPO_KEY_LIBRIBRAIN = "libribrain"
REPO_KEY_LIBRIBRAIN2 = "libribrain2"

REPO_KEYS = (REPO_KEY_LIBRIBRAIN, REPO_KEY_LIBRIBRAIN2)

REPO_KEY_TO_ID = {
    REPO_KEY_LIBRIBRAIN: REPO_LIBRIBRAIN,
    REPO_KEY_LIBRIBRAIN2: REPO_LIBRIBRAIN2,
}


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

DEEP_SUBJECT = "0"

SUBJECTS = tuple(str(i) for i in range(33))                # "0".."32"
NEW_SUBJECTS = tuple(str(i) for i in range(1, 33))         # "1".."32"


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

CORPUS_SHERLOCK = "sherlock"
CORPUS_TIMIT = "timit"
CORPUS_MOCHA = "mocha"
CORPUS_PODCASTS = "podcasts"

CORPORA = (CORPUS_SHERLOCK, CORPUS_TIMIT, CORPUS_MOCHA, CORPUS_PODCASTS)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

PARTITION_TRAIN = "train"
PARTITION_VALIDATION = "validation"
PARTITION_TEST = "test"

PARTITIONS = (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST)


# ---------------------------------------------------------------------------
# Per-corpus tasks (HF folder / BIDS task token)
# ---------------------------------------------------------------------------

# Books in the Sherlock canon and the corresponding BIDS task token.
SHERLOCK_TASKS = tuple(f"Sherlock{i}" for i in range(1, 10))   # Sherlock1..Sherlock9

TIMIT_TASK = "TIMIT"
MOCHATIMIT_TASK = "MOCHATIMIT"
PODCASTS_TASK = "Podcasts"

# Map task token → corpus.
TASK_TO_CORPUS: dict[str, str] = {
    **{t: CORPUS_SHERLOCK for t in SHERLOCK_TASKS},
    TIMIT_TASK: CORPUS_TIMIT,
    MOCHATIMIT_TASK: CORPUS_MOCHA,
    PODCASTS_TASK: CORPUS_PODCASTS,
}


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

CORPUS_ALIASES: dict[str, str] = {
    "sherlock": CORPUS_SHERLOCK,
    "sherlock_holmes": CORPUS_SHERLOCK,
    "timit": CORPUS_TIMIT,
    "mocha": CORPUS_MOCHA,
    "mocha-timit": CORPUS_MOCHA,
    "mochatimit": CORPUS_MOCHA,
    "mocha_timit": CORPUS_MOCHA,
    "podcasts": CORPUS_PODCASTS,
    "podcast": CORPUS_PODCASTS,
    "moth": CORPUS_PODCASTS,
    "the_moth": CORPUS_PODCASTS,
    "themoth": CORPUS_PODCASTS,
}

PARTITION_ALIASES: dict[str, str] = {
    "train": PARTITION_TRAIN,
    "training": PARTITION_TRAIN,
    "val": PARTITION_VALIDATION,
    "valid": PARTITION_VALIDATION,
    "validation": PARTITION_VALIDATION,
    "test": PARTITION_TEST,
    "testing": PARTITION_TEST,
}


# ---------------------------------------------------------------------------
# Default preprocessing string (matches HF folder layout)
# ---------------------------------------------------------------------------

DEFAULT_PREPROCESSING_STR = "bads+headpos+sss+notch+bp+ds"


__all__ = [
    "REPO_LIBRIBRAIN",
    "REPO_LIBRIBRAIN2",
    "REPO_KEY_LIBRIBRAIN",
    "REPO_KEY_LIBRIBRAIN2",
    "REPO_KEYS",
    "REPO_KEY_TO_ID",
    "DEEP_SUBJECT",
    "SUBJECTS",
    "NEW_SUBJECTS",
    "CORPUS_SHERLOCK",
    "CORPUS_TIMIT",
    "CORPUS_MOCHA",
    "CORPUS_PODCASTS",
    "CORPORA",
    "PARTITION_TRAIN",
    "PARTITION_VALIDATION",
    "PARTITION_TEST",
    "PARTITIONS",
    "SHERLOCK_TASKS",
    "TIMIT_TASK",
    "MOCHATIMIT_TASK",
    "PODCASTS_TASK",
    "TASK_TO_CORPUS",
    "CORPUS_ALIASES",
    "PARTITION_ALIASES",
    "DEFAULT_PREPROCESSING_STR",
]
