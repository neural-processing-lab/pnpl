"""
LibriBrain100 — unified loader for LibriBrain + LibriBrain2.

LibriBrain100 is the user-facing PyTorch Dataset wrapping the full
LibriBrain release: a virtual union of the original
``pnpl/LibriBrain`` Hugging Face dataset and the
``pnpl/LibriBrain2`` extension. The deep component (~80h, sub-0)
covers the entire Sherlock Holmes canon plus TIMIT, MOCHA-TIMIT, and
30 'The Moth' podcasts; the broad component (~44 min × 32 subjects)
covers Sherlock book 1 chapters 11 and 12.

Public API:

    from pnpl.datasets import (
        LibriBrain100,
        LibriBrain100Speech,
        LibriBrain100Phoneme,
        LibriBrain100Word,
    )
"""

from __future__ import annotations

from .compat import LibriBrain100Phoneme, LibriBrain100Speech, LibriBrain100Word
from .constants import (
    CORPORA,
    CORPUS_MOCHA,
    CORPUS_PODCASTS,
    CORPUS_SHERLOCK,
    CORPUS_TIMIT,
    DEEP_SUBJECT,
    NEW_SUBJECTS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    PARTITIONS,
    SUBJECTS,
)
from .dataset import LibriBrain100
from .manifest import (
    RUN_KEYS,
    RUN_RECORDS,
    RunRecord,
    TEST_RUN_KEYS,
    VALIDATION_RUN_KEYS,
    get_record,
    select_records,
)
from .selectors import (
    normalize_corpus,
    normalize_partition,
    normalize_subjects,
    validate_selector_combination,
)

__all__ = [
    # Dataset classes
    "LibriBrain100",
    "LibriBrain100Speech",
    "LibriBrain100Phoneme",
    "LibriBrain100Word",
    # Manifest
    "RunRecord",
    "RUN_RECORDS",
    "RUN_KEYS",
    "VALIDATION_RUN_KEYS",
    "TEST_RUN_KEYS",
    "get_record",
    "select_records",
    # Selectors / normalisation
    "normalize_subjects",
    "normalize_corpus",
    "normalize_partition",
    "validate_selector_combination",
    # Constants
    "DEEP_SUBJECT",
    "SUBJECTS",
    "NEW_SUBJECTS",
    "CORPORA",
    "CORPUS_SHERLOCK",
    "CORPUS_TIMIT",
    "CORPUS_MOCHA",
    "CORPUS_PODCASTS",
    "PARTITIONS",
    "PARTITION_TRAIN",
    "PARTITION_VALIDATION",
    "PARTITION_TEST",
]
