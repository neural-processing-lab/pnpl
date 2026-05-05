"""
Manifest of every (subject, session, task, run) record advertised by
LibriBrain100, together with which Hugging Face repo it lives in and
which standard partition it belongs to.

The manifest is the single source of truth for what *should* exist;
``LibriBrain100`` skips records whose underlying files are not
(yet) downloadable, with an informative warning. This keeps the API
stable while the LibriBrain2 upload is in flight.

Layout summary (per the LibriBrain100 paper):

  Subject 0 — deep (~80 h):
    Sherlock1..Sherlock7   pnpl/LibriBrain   (full canon read by D. Clarke)
    Sherlock8..Sherlock9   pnpl/LibriBrain2  (book 8 D. Clarke; book 9 T. Copeland)
    TIMIT                  pnpl/LibriBrain2  (read American English, 630 speakers)
    MOCHATIMIT             pnpl/LibriBrain2  (Southern British English, 2 speakers)
    Podcasts               pnpl/LibriBrain2  (30 'The Moth' stories)

  Subjects 1..32 — broad (~44 min each, multi-subject):
    Sherlock1 ses-11       pnpl/LibriBrain2  (validation)
    Sherlock1 ses-12       pnpl/LibriBrain2  (test)

Standard partitions per the paper's data-splits table (Sec. 3.2):

  - Sherlock (sub-0): all sessions are TRAIN except Sherlock1 ses-11
    (validation) and Sherlock1 ses-12 (test).
  - TIMIT, MOCHATIMIT, Podcasts (sub-0): per-corpus train/val/test
    splits, see ``_split_*`` helpers below.
  - Sherlock1 ses-11/12 (sub-1..32): validation / test respectively.
    Multi-subject data has no train partition by design.

Per-session run numbers and session counts for the not-yet-uploaded
corpora (Sherlock9, TIMIT, Podcasts) are best-effort and may be
updated once the LibriBrain2 upload is finalized; the loader's
``download=True`` path tolerates missing files at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .constants import (
    CORPUS_PODCASTS,
    CORPUS_MOCHA,
    CORPUS_SHERLOCK,
    CORPUS_TIMIT,
    DEEP_SUBJECT,
    MOCHATIMIT_TASK,
    NEW_SUBJECTS,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    PODCASTS_TASK,
    REPO_KEY_LIBRIBRAIN,
    REPO_KEY_LIBRIBRAIN2,
    SHERLOCK_TASKS,
    TASK_TO_CORPUS,
    TIMIT_TASK,
)


@dataclass(frozen=True)
class RunRecord:
    """One BIDS run worth of MEG + events."""

    subject: str
    session: str
    task: str
    run: str
    corpus: str
    repo: str
    partition: str

    @property
    def run_key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.session, self.task, self.run)


# ---------------------------------------------------------------------------
# Sherlock1..Sherlock7 — sub-0, original LibriBrain release.
#
# Mirrors the per-book session/run inventory of the existing
# ``pnpl.datasets.libribrain2025.constants.RUN_KEYS``. We re-declare it
# here rather than importing it because that module fetches a remote
# JSON manifest at import time, which we want to keep this file free
# of (it should be importable even when the network is offline).
# ---------------------------------------------------------------------------

# (session, run) per book, mirroring the actual HF tree of pnpl/LibriBrain
# (verified against the HF tree API; see VM tests
# ``test_manifest_libribrain_records_present_on_hf`` for the live check).
_LIBRIBRAIN_SESSION_RUNS: dict[str, tuple[tuple[str, str], ...]] = {
    "Sherlock1": tuple(
        [(str(i), "1") for i in range(1, 11)]
        + [("11", "2"), ("12", "2")]
    ),
    "Sherlock2": tuple((str(i), "1") for i in range(1, 13)),
    "Sherlock3": tuple((str(i), "1") for i in range(1, 13)),
    "Sherlock4": tuple((str(i), "1") for i in range(1, 13)),
    "Sherlock5": tuple((str(i), "1") for i in range(1, 16)),
    "Sherlock6": tuple((str(i), "1") for i in range(1, 15)),
    "Sherlock7": tuple((str(i), "1") for i in range(1, 15)),
}


def _sherlock_partition(task: str, session: str) -> str:
    """Per the paper: Sherlock1 ses-11 → val, ses-12 → test, rest → train."""
    if task == "Sherlock1" and session == "11":
        return PARTITION_VALIDATION
    if task == "Sherlock1" and session == "12":
        return PARTITION_TEST
    return PARTITION_TRAIN


def _build_libribrain_sherlock_records() -> list[RunRecord]:
    out: list[RunRecord] = []
    for task, session_runs in _LIBRIBRAIN_SESSION_RUNS.items():
        for session, run in session_runs:
            out.append(
                RunRecord(
                    subject=DEEP_SUBJECT,
                    session=session,
                    task=task,
                    run=run,
                    corpus=CORPUS_SHERLOCK,
                    repo=REPO_KEY_LIBRIBRAIN,
                    partition=_sherlock_partition(task, session),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Sherlock8 / Sherlock9 — sub-0, LibriBrain2 release.
#
# The HF tree shows Sherlock8 with ses-1..10 currently. Sherlock9 is
# expected to follow the same one-session-per-LibriVox-track pattern as
# the other books in the canon; we approximate with ses-1..12 and
# tolerate gaps at runtime. Both are entirely train (all-canon Sherlock
# val/test live in book 1 already).
# ---------------------------------------------------------------------------

_LIBRIBRAIN2_SHERLOCK_SESSION_RUNS: dict[str, tuple[tuple[str, str], ...]] = {
    "Sherlock8": tuple((str(i), "1") for i in range(1, 11)),
    # Sherlock9 session count is approximate; will be reconciled with the
    # final upload. Missing sessions are skipped at load time.
    "Sherlock9": tuple((str(i), "1") for i in range(1, 13)),
}


def _build_libribrain2_sherlock_records() -> list[RunRecord]:
    out: list[RunRecord] = []
    for task, session_runs in _LIBRIBRAIN2_SHERLOCK_SESSION_RUNS.items():
        for session, run in session_runs:
            out.append(
                RunRecord(
                    subject=DEEP_SUBJECT,
                    session=session,
                    task=task,
                    run=run,
                    corpus=CORPUS_SHERLOCK,
                    repo=REPO_KEY_LIBRIBRAIN2,
                    partition=PARTITION_TRAIN,
                )
            )
    return out


# ---------------------------------------------------------------------------
# TIMIT — sub-0, LibriBrain2 release.
#
# Per the paper (Sec. 3.2), the standard split follows TIMIT's official
# core test set + Kaldi 50-speaker dev set, applied at the utterance
# level. The MEG-side session count is not yet visible on HF; we
# placeholder with ses-1..10 and assign one session each to val/test.
# Final per-utterance filtering will happen at the events level once
# the upload exposes the per-utterance metadata.
# ---------------------------------------------------------------------------

_TIMIT_TRAIN_SESSIONS = tuple(str(i) for i in range(1, 9))
_TIMIT_VAL_SESSIONS = ("9",)
_TIMIT_TEST_SESSIONS = ("10",)


def _timit_partition(session: str) -> str:
    if session in _TIMIT_VAL_SESSIONS:
        return PARTITION_VALIDATION
    if session in _TIMIT_TEST_SESSIONS:
        return PARTITION_TEST
    return PARTITION_TRAIN


def _build_timit_records() -> list[RunRecord]:
    out: list[RunRecord] = []
    for ses in (
        _TIMIT_TRAIN_SESSIONS + _TIMIT_VAL_SESSIONS + _TIMIT_TEST_SESSIONS
    ):
        out.append(
            RunRecord(
                subject=DEEP_SUBJECT,
                session=ses,
                task=TIMIT_TASK,
                run="1",
                corpus=CORPUS_TIMIT,
                repo=REPO_KEY_LIBRIBRAIN2,
                partition=_timit_partition(ses),
            )
        )
    return out


# ---------------------------------------------------------------------------
# MOCHATIMIT — sub-0, LibriBrain2 release.
#
# 460 sentences × 2 speakers = 920 utterances, organized into four sets
# A..D where sentences repeat between A↔D and B↔C. Per the paper:
# A and D for training, B and C split into equally sized unique-sentence
# subsets for val and test. The HF tree currently shows ses-1..4 which
# we map onto the 4 sets in order.
# ---------------------------------------------------------------------------

_MOCHATIMIT_PARTITION_BY_SESSION: dict[str, str] = {
    "1": PARTITION_TRAIN,        # set A
    "2": PARTITION_VALIDATION,   # set B (split half)
    "3": PARTITION_TEST,         # set C (split half)
    "4": PARTITION_TRAIN,        # set D
}


def _build_mochatimit_records() -> list[RunRecord]:
    return [
        RunRecord(
            subject=DEEP_SUBJECT,
            session=ses,
            task=MOCHATIMIT_TASK,
            run="1",
            corpus=CORPUS_MOCHA,
            repo=REPO_KEY_LIBRIBRAIN2,
            partition=part,
        )
        for ses, part in _MOCHATIMIT_PARTITION_BY_SESSION.items()
    ]


# ---------------------------------------------------------------------------
# Podcasts — sub-0, LibriBrain2 release.
#
# 30 'The Moth' stories, one session per story (per Appendix B.3).
# Following Tang et al. (2023), story 29 ('From Boyhood to Fatherhood')
# is the validation story and story 30 ('Where There's Smoke') is the
# test story; the remaining 28 are training.
# ---------------------------------------------------------------------------

_PODCAST_VAL_SESSIONS = ("29",)
_PODCAST_TEST_SESSIONS = ("30",)


def _podcast_partition(session: str) -> str:
    if session in _PODCAST_VAL_SESSIONS:
        return PARTITION_VALIDATION
    if session in _PODCAST_TEST_SESSIONS:
        return PARTITION_TEST
    return PARTITION_TRAIN


def _build_podcast_records() -> list[RunRecord]:
    return [
        RunRecord(
            subject=DEEP_SUBJECT,
            session=str(i),
            task=PODCASTS_TASK,
            run="1",
            corpus=CORPUS_PODCASTS,
            repo=REPO_KEY_LIBRIBRAIN2,
            partition=_podcast_partition(str(i)),
        )
        for i in range(1, 31)
    ]


# ---------------------------------------------------------------------------
# Subjects 1..32 — multi-subject Sherlock1 ses-11 (val) and ses-12 (test).
#
# Per Sec. 3.2 of the paper, no training partition is assigned to
# multi-subject data by default; supervised-fine-tuning workflows
# typically use the validation partition as their SFT training set.
# ---------------------------------------------------------------------------

def _build_multisubject_records() -> list[RunRecord]:
    out: list[RunRecord] = []
    for subject in NEW_SUBJECTS:
        out.append(
            RunRecord(
                subject=subject,
                session="11",
                task="Sherlock1",
                run="1",
                corpus=CORPUS_SHERLOCK,
                repo=REPO_KEY_LIBRIBRAIN2,
                partition=PARTITION_VALIDATION,
            )
        )
        out.append(
            RunRecord(
                subject=subject,
                session="12",
                task="Sherlock1",
                run="1",
                corpus=CORPUS_SHERLOCK,
                repo=REPO_KEY_LIBRIBRAIN2,
                partition=PARTITION_TEST,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Final manifest
# ---------------------------------------------------------------------------

def _build_run_records() -> tuple[RunRecord, ...]:
    records: list[RunRecord] = []
    records.extend(_build_libribrain_sherlock_records())
    records.extend(_build_libribrain2_sherlock_records())
    records.extend(_build_timit_records())
    records.extend(_build_mochatimit_records())
    records.extend(_build_podcast_records())
    records.extend(_build_multisubject_records())
    # Stable order: subject (numeric), task, session (numeric), run.
    records.sort(
        key=lambda r: (
            int(r.subject) if r.subject.isdigit() else r.subject,
            r.task,
            int(r.session) if r.session.isdigit() else r.session,
            int(r.run) if r.run.isdigit() else r.run,
        )
    )
    return tuple(records)


RUN_RECORDS: tuple[RunRecord, ...] = _build_run_records()


# Convenience tuples: pure run-key views for each partition.
RUN_KEYS: tuple[tuple[str, str, str, str], ...] = tuple(
    r.run_key for r in RUN_RECORDS
)

VALIDATION_RUN_KEYS: tuple[tuple[str, str, str, str], ...] = tuple(
    r.run_key for r in RUN_RECORDS if r.partition == PARTITION_VALIDATION
)

TEST_RUN_KEYS: tuple[tuple[str, str, str, str], ...] = tuple(
    r.run_key for r in RUN_RECORDS if r.partition == PARTITION_TEST
)


_RECORD_BY_RUN_KEY: dict[tuple[str, str, str, str], RunRecord] = {
    r.run_key: r for r in RUN_RECORDS
}


def get_record(run_key: Sequence[str]) -> RunRecord:
    """Look up the manifest record for a 4-tuple run key."""
    key = tuple(map(str, run_key))
    if len(key) != 4:
        raise ValueError(
            f"run_key must be a 4-tuple (subject, session, task, run); got {run_key!r}"
        )
    if key not in _RECORD_BY_RUN_KEY:
        raise KeyError(
            f"Unknown LibriBrain100 run key: {key}. "
            f"See pnpl.datasets.libribrain100.RUN_KEYS for the full list."
        )
    return _RECORD_BY_RUN_KEY[key]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_records(
    *,
    subjects: set[str],
    corpus: set[str],
    partition: Optional[str],
    include_run_keys: Optional[Iterable[Sequence[str]]] = None,
    exclude_run_keys: Optional[Iterable[Sequence[str]]] = None,
    exclude_tasks: Optional[Iterable[str]] = None,
) -> list[RunRecord]:
    """Return manifest records matching the given selectors.

    The selectors compose with set semantics:

      filtered = (
          records
          ∩ {subject ∈ subjects}
          ∩ {corpus  ∈ corpus}
          ∩ {partition == partition}
          ∩ {run_key ∈ include_run_keys}        (if non-empty)
          ∖ {run_key ∈ exclude_run_keys}
          ∖ {task ∈ exclude_tasks}
      )

    Caller is responsible for normalizing ``subjects``, ``corpus``, and
    ``partition`` upstream — this function expects canonical names.
    """
    include_set: set[tuple[str, ...]] = set()
    if include_run_keys:
        include_set = {tuple(map(str, rk)) for rk in include_run_keys}
    exclude_set: set[tuple[str, ...]] = set()
    if exclude_run_keys:
        exclude_set = {tuple(map(str, rk)) for rk in exclude_run_keys}
    exclude_task_set: set[str] = set(exclude_tasks or ())

    overlap = include_set & exclude_set
    if overlap:
        raise ValueError(
            f"Run keys cannot be both included and excluded: {sorted(overlap)}"
        )

    out: list[RunRecord] = []
    for record in RUN_RECORDS:
        if record.subject not in subjects:
            continue
        if record.corpus not in corpus:
            continue
        if partition is not None and record.partition != partition:
            continue
        if include_set and record.run_key not in include_set:
            continue
        if record.run_key in exclude_set:
            continue
        if record.task in exclude_task_set:
            continue
        out.append(record)

    return out


__all__ = [
    "RunRecord",
    "RUN_RECORDS",
    "RUN_KEYS",
    "VALIDATION_RUN_KEYS",
    "TEST_RUN_KEYS",
    "get_record",
    "select_records",
]
