"""Offline tests for the Radboud-backed datasets (Armeni, Schoffelen).

Live WebDAV tests are gated behind ``PNPL_RADBOUD_LIVE=1`` and require
``RADBOUD_USERNAME`` / ``RADBOUD_PASSWORD`` to be set in the
environment. Without those, the tests below cover imports, MRO,
constants, task collect_samples on hand-crafted events, and the
download-disabled error path.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Imports / MRO
# ---------------------------------------------------------------------------

def test_radboud_mixin_imports():
    from pnpl.datasets.mixins import RadboudDownloadMixin

    assert hasattr(RadboudDownloadMixin, "ensure_file")
    assert hasattr(RadboudDownloadMixin, "ensure_directory")
    assert hasattr(RadboudDownloadMixin, "resolve_remote_file")


def test_armeni_mro():
    from pnpl.datasets.mixins import (
        BIDSMixin, ContinuousH5Mixin, RadboudDownloadMixin, StandardizationMixin,
    )
    from pnpl.datasets.armeni2022 import Armeni2022

    for required in (
        RadboudDownloadMixin, StandardizationMixin, ContinuousH5Mixin, BIDSMixin,
    ):
        assert required in Armeni2022.__mro__


def test_schoffelen_mro():
    from pnpl.datasets.mixins import (
        BIDSMixin, ContinuousH5Mixin, RadboudDownloadMixin, StandardizationMixin,
    )
    from pnpl.datasets.schoffelen2019 import Schoffelen2019

    for required in (
        RadboudDownloadMixin, StandardizationMixin, ContinuousH5Mixin, BIDSMixin,
    ):
        assert required in Schoffelen2019.__mro__


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_armeni_constants():
    from pnpl.datasets.armeni2022 import (
        RADBOUD_DATASET_URL, SESSIONS, SUBJECTS, TASKS,
    )
    assert RADBOUD_DATASET_URL.endswith("/DSC_3011085.05_995_v1/")
    assert SUBJECTS == ["001", "002", "003"]
    assert SESSIONS == [f"{i:03d}" for i in range(1, 11)]
    assert TASKS == ["compr"]


def test_schoffelen_constants():
    from pnpl.datasets.schoffelen2019 import (
        CHANNELS, RADBOUD_DATASET_URL, SUBJECTS, TASKS, is_task_for_subject,
    )
    assert RADBOUD_DATASET_URL.endswith("/DSC_3011020.09_236_v1/")
    assert TASKS == ["auditory", "rest", "visual"]
    assert all(s.startswith(("A", "V")) for s in SUBJECTS)
    assert len(CHANNELS) > 200
    assert is_task_for_subject("A2002", "auditory") is True
    assert is_task_for_subject("A2002", "visual") is False
    assert is_task_for_subject("V1001", "visual") is True
    assert is_task_for_subject("V1001", "auditory") is False
    assert is_task_for_subject("V1001", "rest") is True


# ---------------------------------------------------------------------------
# Task: collect_samples on a hand-crafted events file
# ---------------------------------------------------------------------------

def test_armeni_phoneme_task_collect_samples(tmp_path):
    from pnpl.tasks.armeni2022 import PhonemeClassification

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\tsample\ttype\tvalue\n"
        # phoneme rows (mix of position suffixes, stress digits, and oov values)
        "0.10\t0.05\t100\tphoneme_onset_01\t\"AH0\"\n"
        "0.20\t0.05\t200\tphoneme_onset_02\t\"IY1\"\n"
        "0.30\t0.05\t300\tphoneme_onset_01\t\"sp\"\n"        # not in ARPABET → drop
        "0.40\t0.05\t400\tphoneme_onset_01\t\"D\"\n"
        # word + wav rows should be ignored
        "0.50\t0.20\t500\tword_onset_01\t\"THE\"\n"
        "0.00\t0.00\t0\twav_onset\t\"x\"\n"
        # negative-onset row dropped by skip_negative_onset
        "-0.50\t0.05\t-500\tphoneme_onset_01\t\"AE0\"\n"
    )

    class FakeDataset:
        run_keys = [("001", "001", "compr", "01")]

        def get_events_path(self, *a, **k):
            return str(events_path)

    task = PhonemeClassification(label_type="phoneme")
    samples = task.collect_samples(FakeDataset())
    arpa = [s[5] for s in samples]
    assert arpa == ["AH", "IY", "D"]


def test_armeni_phoneme_voicing_lookup():
    from pnpl.tasks.armeni2022 import PhonemeClassification

    task = PhonemeClassification(label_type="voicing")
    info = task.label_info
    assert info["classes"] == ["uv", "v"]
    # AH is voiced, T is voiceless
    sample_v = ("001", "001", "compr", "01", 0.1, "AH")
    sample_uv = ("001", "001", "compr", "01", 0.1, "T")
    assert task.get_label(sample_v) == 1
    assert task.get_label(sample_uv) == 0


def test_schoffelen_trial_task_collect_samples(tmp_path):
    from pnpl.tasks.schoffelen2019 import TrialEpoching

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\tsample\ttype\tvalue\n"
        "0\t10\t1\ttrial\tn/a\n"
        "0.5\tn/a\t500\tUPPT001\t10\n"
        "10\t10\t10001\ttrial\tn/a\n"
        "10.5\tn/a\t10500\tUPPT001\t20\n"
        "20\t10\t20001\ttrial\tn/a\n"
        # No UPPT001 after this trial → "trigger:none"
        "30\t10\t30001\ttrial\tn/a\n"
    )

    class FakeDataset:
        run_keys = [("A2002", "01", "auditory", "01")]

        def get_events_path(self, *a, **k):
            return str(events_path)

    task = TrialEpoching(tmin=0.0, tmax=1.0, label_type="trigger")
    samples = task.collect_samples(FakeDataset())
    labels = [s[5] for s in samples]
    assert labels == ["trigger:10", "trigger:20", "trigger:none", "trigger:none"]
    assert task.label_info["n_classes"] == 3  # 10, 20, none


# ---------------------------------------------------------------------------
# Construction with download disabled
# ---------------------------------------------------------------------------

def test_armeni_no_local_data_raises(tmp_path):
    from pnpl.datasets.armeni2022 import Armeni2022
    from pnpl.tasks.armeni2022 import PhonemeClassification

    with pytest.raises((FileNotFoundError, ValueError)):
        Armeni2022(
            data_path=str(tmp_path / "nope"),
            task=PhonemeClassification(),
            include_subjects=["001"],
            include_sessions=["001"],
            include_tasks=["compr"],
            download=False,
            create_h5_if_missing=False,
        )


def test_schoffelen_no_local_data_raises(tmp_path):
    from pnpl.datasets.schoffelen2019 import Schoffelen2019
    from pnpl.tasks.schoffelen2019 import TrialEpoching

    with pytest.raises((FileNotFoundError, ValueError)):
        Schoffelen2019(
            data_path=str(tmp_path / "nope"),
            task=TrialEpoching(),
            include_subjects=["A2002"],
            include_tasks=["auditory"],
            download=False,
            create_h5_if_missing=False,
        )


# ---------------------------------------------------------------------------
# Live WebDAV (skipped without PNPL_RADBOUD_LIVE)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (
        os.getenv("PNPL_RADBOUD_LIVE")
        and os.getenv("RADBOUD_USERNAME")
        and os.getenv("RADBOUD_PASSWORD")
    ),
    reason="Set PNPL_RADBOUD_LIVE=1 + RADBOUD_USERNAME/PASSWORD to run live WebDAV tests.",
)
def test_armeni_webdav_listing_lazy():
    """Confirm WebDAV listing works for Armeni — no downloads."""
    from pnpl.datasets.armeni2022 import Armeni2022

    root = Armeni2022._list_folder("")
    names = {e["rel_path"] for e in root}
    for expected in ("sub-001", "sub-002", "sub-003", "README.txt"):
        assert expected in names


@pytest.mark.skipif(
    not (
        os.getenv("PNPL_RADBOUD_LIVE")
        and os.getenv("RADBOUD_USERNAME")
        and os.getenv("RADBOUD_PASSWORD")
    ),
    reason="Set PNPL_RADBOUD_LIVE=1 + RADBOUD_USERNAME/PASSWORD to run live WebDAV tests.",
)
def test_schoffelen_webdav_listing_lazy():
    from pnpl.datasets.schoffelen2019 import Schoffelen2019

    root = Schoffelen2019._list_folder("")
    names = {e["rel_path"] for e in root}
    # at least a few representative subject folders
    assert "sub-A2002" in names
    assert "sub-V1001" in names
