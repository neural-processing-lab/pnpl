"""Rudimentary tests for the LibriBrain100 loader.

These cover:
  - imports / public exports
  - selector normalization (subjects / corpus / partition + aliases)
  - manifest invariants (partition disjointness, expected counts,
    multi-subject layout, repo assignment)
  - validation rules raised by the selector layer
  - download-disabled error path

A comprehensive suite that hits the real HF repos lives on the VM at
``/workspace/libribrain100-tests/``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Imports / MRO / public surface
# ---------------------------------------------------------------------------

def test_libribrain100_public_imports():
    from pnpl.datasets import (
        LibriBrain100,
        LibriBrain100Phoneme,
        LibriBrain100Speech,
        LibriBrain100Word,
    )
    assert LibriBrain100 is not None
    assert LibriBrain100Speech is not None
    assert LibriBrain100Phoneme is not None
    assert LibriBrain100Word is not None


def test_libribrain100_mro():
    from pnpl.datasets.libribrain100 import LibriBrain100
    from pnpl.datasets.mixins import (
        BIDSMixin,
        ContinuousH5Mixin,
        HFDownloadMixin,
        StandardizationMixin,
    )
    for required in (
        HFDownloadMixin, StandardizationMixin, ContinuousH5Mixin, BIDSMixin,
    ):
        assert required in LibriBrain100.__mro__, f"{required.__name__} missing"


# ---------------------------------------------------------------------------
# Selector normalization
# ---------------------------------------------------------------------------

def test_normalize_subjects_aliases():
    from pnpl.datasets.libribrain100 import normalize_subjects
    assert normalize_subjects("all") == {str(i) for i in range(33)}
    assert normalize_subjects("new") == {str(i) for i in range(1, 33)}
    assert normalize_subjects("broad") == {str(i) for i in range(1, 33)}
    assert normalize_subjects("deep") == {"0"}
    assert normalize_subjects(0) == {"0"}
    assert normalize_subjects("0") == {"0"}
    assert normalize_subjects("sub-0") == {"0"}
    assert normalize_subjects("sub-07") == {"7"}
    assert normalize_subjects([1, 2, 3]) == {"1", "2", "3"}
    assert normalize_subjects(range(1, 5)) == {"1", "2", "3", "4"}


def test_normalize_subjects_rejects_unknown():
    from pnpl.datasets.libribrain100 import normalize_subjects
    with pytest.raises(ValueError):
        normalize_subjects("not-a-subject")
    with pytest.raises(ValueError):
        normalize_subjects(-1)
    with pytest.raises(ValueError):
        normalize_subjects([])


def test_normalize_corpus_aliases():
    from pnpl.datasets.libribrain100 import normalize_corpus
    assert normalize_corpus("all") == {"sherlock", "timit", "mocha", "podcasts"}
    assert normalize_corpus("sherlock") == {"sherlock"}
    assert normalize_corpus("timit") == {"timit"}
    assert normalize_corpus("mocha") == {"mocha"}
    assert normalize_corpus("mocha-timit") == {"mocha"}
    assert normalize_corpus("mochatimit") == {"mocha"}
    assert normalize_corpus("podcasts") == {"podcasts"}
    assert normalize_corpus("the_moth") == {"podcasts"}
    assert normalize_corpus("moth") == {"podcasts"}
    assert normalize_corpus(["timit", "mocha"]) == {"timit", "mocha"}


def test_normalize_corpus_rejects_unknown():
    from pnpl.datasets.libribrain100 import normalize_corpus
    with pytest.raises(ValueError):
        normalize_corpus("klingon")
    with pytest.raises(ValueError):
        normalize_corpus(["sherlock", "klingon"])
    with pytest.raises(ValueError):
        normalize_corpus([])


def test_normalize_partition_aliases():
    from pnpl.datasets.libribrain100 import normalize_partition
    assert normalize_partition(None) is None
    assert normalize_partition("train") == "train"
    assert normalize_partition("val") == "validation"
    assert normalize_partition("valid") == "validation"
    assert normalize_partition("validation") == "validation"
    assert normalize_partition("Test") == "test"
    with pytest.raises(ValueError):
        normalize_partition("dev")


# ---------------------------------------------------------------------------
# Selector validation
# ---------------------------------------------------------------------------

def test_validate_rejects_new_with_train():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        validate_selector_combination,
    )
    with pytest.raises(ValueError, match="train partition"):
        validate_selector_combination(
            subjects=normalize_subjects("new"),
            corpus=normalize_corpus("sherlock"),
            partition="train",
        )


def test_validate_rejects_new_with_non_sherlock():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        validate_selector_combination,
    )
    with pytest.raises(ValueError, match="non-deep"):
        validate_selector_combination(
            subjects=normalize_subjects("new"),
            corpus=normalize_corpus("timit"),
            partition=None,
        )


def test_validate_allows_all_with_train():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        validate_selector_combination,
    )
    # Subject 0 is included → 'train' is valid.
    validate_selector_combination(
        subjects=normalize_subjects("all"),
        corpus=normalize_corpus("all"),
        partition="train",
    )


# ---------------------------------------------------------------------------
# Manifest invariants
# ---------------------------------------------------------------------------

def test_manifest_partitions_are_disjoint():
    from pnpl.datasets.libribrain100 import RUN_RECORDS

    train = {r.run_key for r in RUN_RECORDS if r.partition == "train"}
    val = {r.run_key for r in RUN_RECORDS if r.partition == "validation"}
    test = {r.run_key for r in RUN_RECORDS if r.partition == "test"}

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == len(RUN_RECORDS)


def test_manifest_run_keys_unique():
    from pnpl.datasets.libribrain100 import RUN_RECORDS

    keys = [r.run_key for r in RUN_RECORDS]
    assert len(keys) == len(set(keys)), "duplicate run keys in manifest"


def test_manifest_subject_zero_has_all_corpora():
    from pnpl.datasets.libribrain100 import CORPORA, RUN_RECORDS

    sub_zero_corpora = {r.corpus for r in RUN_RECORDS if r.subject == "0"}
    assert sub_zero_corpora == set(CORPORA), (
        f"sub-0 missing corpora: {set(CORPORA) - sub_zero_corpora}"
    )


def test_manifest_multisubject_layout():
    """Each of sub-1..32 has exactly Sherlock1 ses-11 (val) + ses-12 (test)."""
    from pnpl.datasets.libribrain100 import NEW_SUBJECTS, RUN_RECORDS

    for subject in NEW_SUBJECTS:
        recs = [r for r in RUN_RECORDS if r.subject == subject]
        assert len(recs) == 2, f"sub-{subject} should have 2 runs, got {len(recs)}"
        partitions = {r.partition for r in recs}
        assert partitions == {"validation", "test"}, (
            f"sub-{subject} partitions wrong: {partitions}"
        )
        tasks = {r.task for r in recs}
        assert tasks == {"Sherlock1"}, (
            f"sub-{subject} tasks wrong: {tasks}"
        )
        sessions = {r.session for r in recs}
        assert sessions == {"11", "12"}, (
            f"sub-{subject} sessions wrong: {sessions}"
        )


def test_manifest_multisubject_has_no_train():
    from pnpl.datasets.libribrain100 import NEW_SUBJECTS, RUN_RECORDS

    train_for_new = [
        r for r in RUN_RECORDS
        if r.subject in NEW_SUBJECTS and r.partition == "train"
    ]
    assert train_for_new == [], (
        "Multi-subject runs must not have train partition by default"
    )


def test_manifest_repo_assignment():
    """Sherlock1..7 → libribrain; everything else → libribrain2."""
    from pnpl.datasets.libribrain100 import RUN_RECORDS

    libribrain_tasks = {f"Sherlock{i}" for i in range(1, 8)}
    for record in RUN_RECORDS:
        if record.subject == "0" and record.task in libribrain_tasks:
            assert record.repo == "libribrain", (
                f"{record.run_key} should be in libribrain, got {record.repo}"
            )
        else:
            assert record.repo == "libribrain2", (
                f"{record.run_key} should be in libribrain2, got {record.repo}"
            )


def test_get_record_lookup():
    from pnpl.datasets.libribrain100 import RUN_KEYS, get_record

    rk = RUN_KEYS[0]
    record = get_record(rk)
    assert record.run_key == rk

    with pytest.raises(KeyError):
        get_record(("0", "999", "Sherlock1", "1"))

    with pytest.raises(ValueError):
        get_record(("0", "1"))


# ---------------------------------------------------------------------------
# select_records semantics
# ---------------------------------------------------------------------------

def test_select_records_partition_filter():
    from pnpl.datasets.libribrain100 import (
        VALIDATION_RUN_KEYS,
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    recs = select_records(
        subjects=normalize_subjects("all"),
        corpus=normalize_corpus("all"),
        partition="validation",
    )
    assert {r.run_key for r in recs} == set(VALIDATION_RUN_KEYS)


def test_select_records_subject_filter():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    recs = select_records(
        subjects=normalize_subjects("deep"),
        corpus=normalize_corpus("all"),
        partition=None,
    )
    assert {r.subject for r in recs} == {"0"}


def test_select_records_corpus_filter():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    recs = select_records(
        subjects=normalize_subjects("all"),
        corpus=normalize_corpus("podcasts"),
        partition=None,
    )
    assert all(r.corpus == "podcasts" for r in recs)
    # Per the manifest there are 30 'The Moth' stories, all sub-0.
    assert len(recs) == 30


def test_select_records_include_run_keys():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    rk = ("0", "11", "Sherlock1", "2")
    recs = select_records(
        subjects=normalize_subjects("all"),
        corpus=normalize_corpus("all"),
        partition=None,
        include_run_keys=[rk],
    )
    assert [r.run_key for r in recs] == [rk]


def test_select_records_exclude_tasks():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    recs = select_records(
        subjects=normalize_subjects("deep"),
        corpus=normalize_corpus("sherlock"),
        partition=None,
        exclude_tasks=["Sherlock1"],
    )
    assert all(r.task != "Sherlock1" for r in recs)


def test_select_records_overlap_raises():
    from pnpl.datasets.libribrain100 import (
        normalize_corpus,
        normalize_subjects,
        select_records,
    )
    rk = ("0", "11", "Sherlock1", "2")
    with pytest.raises(ValueError, match="both included and excluded"):
        select_records(
            subjects=normalize_subjects("all"),
            corpus=normalize_corpus("all"),
            partition=None,
            include_run_keys=[rk],
            exclude_run_keys=[rk],
        )


# ---------------------------------------------------------------------------
# Construction with download disabled
# ---------------------------------------------------------------------------

def test_no_local_data_raises(tmp_path):
    from pnpl.datasets.libribrain100 import LibriBrain100
    from pnpl.tasks import SpeechDetection

    with pytest.raises((FileNotFoundError, ValueError)):
        LibriBrain100(
            data_path=str(tmp_path / "nope"),
            task=SpeechDetection(tmin=0.0, tmax=0.5),
            partition="validation",
            subjects="deep",
            corpus="sherlock",
            download=False,
            preload_files=False,
        )


def test_partition_and_explicit_run_keys_raises(tmp_path):
    from pnpl.datasets.libribrain100 import LibriBrain100
    from pnpl.tasks import SpeechDetection

    with pytest.raises(ValueError, match="shortcut"):
        LibriBrain100(
            data_path=str(tmp_path / "nope"),
            task=SpeechDetection(tmin=0.0, tmax=0.5),
            partition="train",
            include_run_keys=[("0", "1", "Sherlock1", "1")],
            download=False,
        )
