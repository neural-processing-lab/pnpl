"""
User-facing selector normalization for LibriBrain100.

Lets the dataset accept a small, ergonomic set of inputs for
``subjects``, ``corpus``, and ``partition`` and turns them into the
canonical sets the manifest is built around.

Validation lives here so manifest filtering can stay a pure
set-intersection.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence, Union

from .constants import (
    CORPORA,
    CORPUS_ALIASES,
    CORPUS_SHERLOCK,
    DEEP_SUBJECT,
    NEW_SUBJECTS,
    PARTITION_ALIASES,
    PARTITION_TRAIN,
    PARTITIONS,
    SUBJECTS,
)


SubjectsArg = Union[str, int, Sequence[Union[str, int]], range, None]
CorpusArg = Union[str, Sequence[str], None]
PartitionArg = Union[str, None]


_SUBJECT_PATTERN = re.compile(r"(?:sub-)?0*(\d+)$")


def _coerce_subject_id(value) -> str:
    """Normalize a single subject id to its canonical string form."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Subject ids must be non-negative; got {value}")
        return str(value)
    if isinstance(value, str):
        match = _SUBJECT_PATTERN.match(value.strip())
        if match is None:
            raise ValueError(f"Could not parse subject id: {value!r}")
        return str(int(match.group(1)))
    raise ValueError(
        f"Subject ids must be int or str, got {type(value).__name__}: {value!r}"
    )


def normalize_subjects(subjects: SubjectsArg) -> set[str]:
    """Compile a ``subjects=...`` argument into a set of canonical ids.

    Accepted forms:
      - ``"all"`` — every subject (sub-0 + sub-1..32)
      - ``"deep"`` — sub-0 (the deep single-subject component)
      - ``"broad"`` — sub-1..32 (the broad multi-subject component)
      - ``0`` / ``"0"`` / ``"sub-0"`` — single subject
      - any iterable / range of ints or string ids
    """
    if subjects is None or subjects == "all":
        return set(SUBJECTS)
    if isinstance(subjects, str):
        token = subjects.strip().lower()
        if token == "all":
            return set(SUBJECTS)
        if token == "broad":
            return set(NEW_SUBJECTS)
        if token == "deep":
            return {DEEP_SUBJECT}
        return {_coerce_subject_id(subjects)}
    if isinstance(subjects, int):
        return {_coerce_subject_id(subjects)}
    if isinstance(subjects, range):
        return {_coerce_subject_id(i) for i in subjects}
    if isinstance(subjects, Iterable):
        out = {_coerce_subject_id(s) for s in subjects}
        if not out:
            raise ValueError("subjects=[] is empty; pass 'all' or specific ids")
        return out
    raise ValueError(f"Unsupported subjects argument: {subjects!r}")


def normalize_corpus(corpus: CorpusArg) -> set[str]:
    """Compile a ``corpus=...`` argument into a set of canonical names."""
    if corpus is None or corpus == "all":
        return set(CORPORA)
    if isinstance(corpus, str):
        token = corpus.strip().lower()
        if token == "all":
            return set(CORPORA)
        if token not in CORPUS_ALIASES:
            raise ValueError(
                f"Unknown corpus: {corpus!r}. "
                f"Known: {sorted(set(CORPUS_ALIASES.keys()))}"
            )
        return {CORPUS_ALIASES[token]}
    if isinstance(corpus, Iterable):
        out: set[str] = set()
        for c in corpus:
            if not isinstance(c, str):
                raise ValueError(
                    f"corpus list entries must be strings, got {c!r}"
                )
            token = c.strip().lower()
            if token not in CORPUS_ALIASES:
                raise ValueError(
                    f"Unknown corpus: {c!r}. "
                    f"Known: {sorted(set(CORPUS_ALIASES.keys()))}"
                )
            out.add(CORPUS_ALIASES[token])
        if not out:
            raise ValueError("corpus=[] is empty; pass 'all' or specific names")
        return out
    raise ValueError(f"Unsupported corpus argument: {corpus!r}")


def normalize_partition(partition: PartitionArg) -> str | None:
    """Compile a ``partition=...`` argument into the canonical name (or None)."""
    if partition is None:
        return None
    if not isinstance(partition, str):
        raise ValueError(
            f"partition must be a string or None; got {type(partition).__name__}"
        )
    token = partition.strip().lower()
    if token not in PARTITION_ALIASES:
        raise ValueError(
            f"Unknown partition: {partition!r}. Must be one of {PARTITIONS} "
            f"(aliases: 'val'/'valid')."
        )
    return PARTITION_ALIASES[token]


def validate_selector_combination(
    *,
    subjects: set[str],
    corpus: set[str],
    partition: str | None,
) -> None:
    """Raise if the requested combination is empty by construction.

    LibriBrain100 has known data-shape constraints we want to surface
    early, before any download is attempted:

    - subjects ⊆ {1..32} only listened to Sherlock; selecting them with
      a non-Sherlock corpus yields no records.
    - subjects ⊆ {1..32} have no train data per the standard splits;
      ``partition='train'`` with only those subjects yields no records.
      Use ``partition='validation'`` as an SFT training set if needed.
    """
    only_new = bool(subjects) and subjects.isdisjoint({DEEP_SUBJECT})

    if only_new and corpus != {CORPUS_SHERLOCK}:
        non_sherlock = sorted(corpus - {CORPUS_SHERLOCK})
        raise ValueError(
            f"subjects={sorted(subjects)} contains only multi-subject "
            f"(non-deep) ids, which only have Sherlock recordings. "
            f"Requested corpus={non_sherlock} is not available for these "
            f"subjects. Either include subject 0 in subjects or restrict "
            f"to corpus='sherlock'."
        )

    if only_new and partition == PARTITION_TRAIN:
        raise ValueError(
            "subjects='broad' (or any selection without subject 0) has no "
            "train partition by design — sub-1..32 contribute Sherlock1 "
            "ses-11 (validation) and ses-12 (test) only. For a "
            "supervised-fine-tuning workflow on the broad subjects, "
            "use partition='validation' as the SFT training set and "
            "partition='test' for evaluation."
        )


__all__ = [
    "SubjectsArg",
    "CorpusArg",
    "PartitionArg",
    "normalize_subjects",
    "normalize_corpus",
    "normalize_partition",
    "validate_selector_combination",
]
