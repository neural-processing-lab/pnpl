"""Offline tests for the Pallier2025 (OpenNeuro ds007523) loader.

Live network tests are gated behind ``PNPL_OPENNEURO_LIVE=1``. Without
that, these cover imports, MRO, constants, and the WordDetection task
running against a hand-crafted events.tsv.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Imports / MRO
# ---------------------------------------------------------------------------

def test_openneuro_mixin_imports():
    from pnpl.datasets.mixins import OpenNeuroDownloadMixin

    assert hasattr(OpenNeuroDownloadMixin, "ensure_file")
    assert hasattr(OpenNeuroDownloadMixin, "resolve_remote_file")
    assert hasattr(OpenNeuroDownloadMixin, "list_remote_files")


def test_pallier2025_mro():
    from pnpl.datasets.mixins import (
        BIDSMixin, ContinuousH5Mixin, OpenNeuroDownloadMixin, StandardizationMixin,
    )
    from pnpl.datasets.pallier2025 import Pallier2025

    for required in (
        OpenNeuroDownloadMixin, StandardizationMixin, ContinuousH5Mixin, BIDSMixin,
    ):
        assert required in Pallier2025.__mro__


def test_pallier2025_publicly_exposed():
    from pnpl.datasets import Pallier2025
    from pnpl.datasets.pallier2025.dataset import Pallier2025 as _direct

    assert Pallier2025 is _direct


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_pallier2025_constants():
    from pnpl.datasets.pallier2025 import Pallier2025
    from pnpl.datasets.pallier2025.constants import (
        OPENNEURO_DATASET_ID,
        OPENNEURO_SNAPSHOT_TAG,
        RUNS,
        SESSIONS,
        SUBJECTS,
        TASKS,
    )

    assert OPENNEURO_DATASET_ID == "ds007523"
    assert OPENNEURO_SNAPSHOT_TAG == "1.0.1"
    assert SUBJECTS == [f"{i:02d}" for i in range(1, 59)]
    assert len(SUBJECTS) == 58
    assert SESSIONS == ["01"]
    assert TASKS == ["listen"]
    assert RUNS == [f"{i:02d}" for i in range(1, 10)]
    assert Pallier2025.OPENNEURO_DATASET_ID == "ds007523"
    assert Pallier2025.OPENNEURO_SNAPSHOT_TAG == "1.0.1"


# ---------------------------------------------------------------------------
# OpenNeuro URL construction
# ---------------------------------------------------------------------------

def test_openneuro_url_join():
    from pnpl.datasets.mixins import OpenNeuroDownloadMixin

    class _Probe(OpenNeuroDownloadMixin):
        OPENNEURO_DATASET_ID = "ds007523"

    url = _Probe._join_url("sub-01/ses-01/meg/sub-01_ses-01_task-listen_run-01_meg.fif")
    assert url == (
        "https://s3.amazonaws.com/openneuro.org/ds007523/"
        "sub-01/ses-01/meg/sub-01_ses-01_task-listen_run-01_meg.fif"
    )


# ---------------------------------------------------------------------------
# Task: collect_samples on a hand-crafted events file
# ---------------------------------------------------------------------------

def test_word_detection_collect_samples(tmp_path):
    from pnpl.tasks.pallier2025 import WordDetection

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tstimulus\n"
        "1.234\t0.184\tWord\tlorsque\n"
        "1.418\t0.062\tWord\tj\n"           # short, kept by default
        "1.480\t0.150\tWord\tavais\n"
        "2.000\t0.120\tWord\tLorsque\n"     # case-folded → 'lorsque'
        "3.000\t0.000\tOther\tnoise\n"      # non-word row dropped
        "bad\t0.10\tWord\tfoo\n"            # non-numeric onset dropped
    )

    class FakeDataset:
        run_keys = [("01", "01", "listen", "01")]

        def get_events_path(self, *a, **k):
            return str(events_path)

    task = WordDetection(tmin=0.0, tmax=3.0)
    samples = task.collect_samples(FakeDataset())
    words = [s[5] for s in samples]
    assert words == ["lorsque", "j", "avais", "lorsque"]
    assert task.label_info["classes"] == ["avais", "j", "lorsque"]
    assert task.label_info["n_classes"] == 3


def test_word_detection_min_word_length_filter(tmp_path):
    from pnpl.tasks.pallier2025 import WordDetection

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tstimulus\n"
        "1.0\t0.1\tWord\tj\n"
        "1.5\t0.1\tWord\ta\n"
        "2.0\t0.1\tWord\tlorsque\n"
        "2.5\t0.1\tWord\tavais\n"
    )

    class FakeDataset:
        run_keys = [("01", "01", "listen", "01")]

        def get_events_path(self, *a, **k):
            return str(events_path)

    task = WordDetection(min_word_length=2)
    samples = task.collect_samples(FakeDataset())
    words = [s[5] for s in samples]
    assert "j" not in words
    assert "a" not in words
    assert set(words) == {"lorsque", "avais"}


def test_word_detection_keep_top_k(tmp_path):
    from pnpl.tasks.pallier2025 import WordDetection

    events_path = tmp_path / "events.tsv"
    # 'lorsque' x3, 'j' x2, 'avais' x1 — top-2 should drop 'avais'
    events_path.write_text(
        "onset\tduration\ttrial_type\tstimulus\n"
        "1.0\t0.1\tWord\tlorsque\n"
        "1.1\t0.1\tWord\tlorsque\n"
        "1.2\t0.1\tWord\tlorsque\n"
        "1.3\t0.1\tWord\tj\n"
        "1.4\t0.1\tWord\tj\n"
        "1.5\t0.1\tWord\tavais\n"
    )

    class FakeDataset:
        run_keys = [("01", "01", "listen", "01")]

        def get_events_path(self, *a, **k):
            return str(events_path)

    task = WordDetection(keep_top_k=2)
    samples = task.collect_samples(FakeDataset())
    words = sorted({s[5] for s in samples})
    assert words == ["j", "lorsque"]
    assert task.label_info["n_classes"] == 2


# ---------------------------------------------------------------------------
# Construction with download disabled
# ---------------------------------------------------------------------------

def test_pallier2025_no_local_data_raises(tmp_path):
    from pnpl.datasets.pallier2025 import Pallier2025
    from pnpl.tasks.pallier2025 import WordDetection

    with pytest.raises((FileNotFoundError, ValueError)):
        Pallier2025(
            data_path=str(tmp_path / "nope"),
            task=WordDetection(),
            include_subjects=["01"],
            include_runs=["01"],
            download=False,
            create_h5_if_missing=False,
        )


# ---------------------------------------------------------------------------
# Live network probe (skipped without PNPL_OPENNEURO_LIVE)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("PNPL_OPENNEURO_LIVE"),
    reason="Set PNPL_OPENNEURO_LIVE=1 to run the live ds007523 HEAD probe.",
)
def test_pallier2025_resolve_remote_file_live():
    from pnpl.datasets.pallier2025 import Pallier2025

    info = Pallier2025.resolve_remote_file(
        "sub-01/ses-01/meg/sub-01_ses-01_task-listen_run-01_events.tsv"
    )
    assert info["size"] is not None and info["size"] > 0
    assert info["url"].endswith(
        "sub-01/ses-01/meg/sub-01_ses-01_task-listen_run-01_events.tsv"
    )
