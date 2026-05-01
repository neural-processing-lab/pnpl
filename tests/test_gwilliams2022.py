"""Tests for the rewritten MEG-MASC (Gwilliams et al., 2022) dataset.

Most tests are network-free: they exercise imports, constants, and
download-disabled construction. The OSF manifest test is gated behind a
``PNPL_OSF_LIVE`` env var so the suite stays fast offline.
"""

from __future__ import annotations

import importlib
import os
import re

import pytest


def test_gwilliams_imports_and_mro():
    from pnpl.datasets.mixins import (
        BIDSMixin,
        ContinuousH5Mixin,
        OSFDownloadMixin,
        StandardizationMixin,
    )
    from pnpl.datasets.gwilliams2022 import Gwilliams2022

    mro = Gwilliams2022.__mro__
    for required in (
        OSFDownloadMixin,
        StandardizationMixin,
        ContinuousH5Mixin,
        BIDSMixin,
    ):
        assert required in mro, f"Gwilliams2022 should inherit from {required.__name__}"


def test_constants_match_paper():
    from pnpl.datasets.gwilliams2022 import (
        OSF_PROJECT_ID,
        OSF_PROJECT_FALLBACKS,
        SESSIONS,
        SUBJECTS,
        TASKS,
        TASK_STORIES,
    )

    assert OSF_PROJECT_ID == "ag3kj"
    assert OSF_PROJECT_FALLBACKS == ["h2tzn", "u5327", "dr4wy"]
    assert SUBJECTS == [f"{i:02d}" for i in range(1, 28)]
    assert SESSIONS == ["0", "1"]
    assert TASKS == ["0", "1", "2", "3"]
    assert set(TASK_STORIES) == set(TASKS)


def test_phoneme_task_label_info_voicing():
    from pnpl.tasks.gwilliams2022 import PhonemeClassification

    task = PhonemeClassification(label_type="voicing")
    info = task.label_info
    assert info["classes"] == ["uv", "v"]
    assert info["n_classes"] == 2


def test_phoneme_task_invalid_label_type():
    from pnpl.tasks.gwilliams2022 import PhonemeClassification

    with pytest.raises(ValueError):
        PhonemeClassification(label_type="bogus")


def test_phoneme_task_label_lookup_after_collect(tmp_path):
    """Standalone collect_samples test using a hand-crafted events file —
    avoids any OSF or local raw data dependency."""
    from pnpl.tasks.gwilliams2022 import PhonemeClassification

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0.10\t0.05\t{'kind': 'phoneme', 'phoneme': 't_B', 'pronounced': 1.0}\t1\t100\n"
        "0.20\t0.05\t{'kind': 'phoneme', 'phoneme': 'eh_I', 'pronounced': 1.0}\t1\t200\n"
        "0.30\t0.05\t{'kind': 'phoneme', 'phoneme': 'h#_S', 'pronounced': 1.0}\t1\t300\n"
        "0.40\t0.05\t{'kind': 'word', 'word': 'tara', 'pronounced': 1.0}\t1\t400\n"
        "0.50\t0.05\t{'kind': 'phoneme', 'phoneme': 'r_E', 'pronounced': 0.0}\t1\t500\n"
    )

    class FakeDataset:
        run_keys = [("01", "0", "0", "01")]

        def get_events_path(self, *args, **kwargs):
            return str(events_path)

    task = PhonemeClassification(label_type="phoneme")
    samples = task.collect_samples(FakeDataset())

    # Two phonemes survive: 't_B' -> 'T', 'eh_I' -> 'EH'. 'h#' maps to 'SIL'
    # (dropped). 'word' kind is skipped. 'r_E' is dropped (pronounced=0).
    arpa = [s[5] for s in samples]
    assert arpa == ["T", "EH"]
    assert task.label_info["n_classes"] == 39  # ARPABET (no SIL)
    # Encoded label IDs round-trip through the task.
    label_to_id = task.label_info["label_to_id"]
    assert task.get_label(samples[0]) == label_to_id["T"]
    assert task.get_label(samples[1]) == label_to_id["EH"]


def test_word_task_label_lookup_after_collect(tmp_path):
    from pnpl.tasks.gwilliams2022 import WordDetection

    events_path = tmp_path / "events.tsv"
    events_path.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0.10\t0.30\t{'kind': 'word', 'word': 'Tara', 'pronounced': 1.0}\t1\t100\n"
        "0.50\t0.30\t{'kind': 'word', 'word': 'walked', 'pronounced': 1.0}\t1\t500\n"
        "0.90\t0.30\t{'kind': 'phoneme', 'phoneme': 't_B', 'pronounced': 1.0}\t1\t900\n"
    )

    class FakeDataset:
        run_keys = [("01", "0", "0", "01")]

        def get_events_path(self, *args, **kwargs):
            return str(events_path)

    task = WordDetection()
    samples = task.collect_samples(FakeDataset())
    assert [s[5] for s in samples] == ["tara", "walked"]
    assert task.label_info["classes"] == ["tara", "walked"]


def test_construction_without_download_raises_for_unknown_data():
    from pnpl.datasets.gwilliams2022 import Gwilliams2022
    from pnpl.tasks.gwilliams2022 import PhonemeClassification

    # download=False + non-existent data_path → no run keys discoverable.
    with pytest.raises(ValueError, match="No MEG-MASC run keys"):
        Gwilliams2022(
            data_path="/tmp/__nonexistent_meg_masc__",
            task=PhonemeClassification(),
            download=False,
        )


@pytest.mark.skipif(
    not os.getenv("PNPL_OSF_LIVE"),
    reason="Set PNPL_OSF_LIVE=1 to run live OSF manifest tests.",
)
def test_osf_manifest_lists_all_subjects():
    """End-to-end test that hits the OSF API. Off by default to keep CI cheap."""
    from pnpl.datasets.gwilliams2022 import Gwilliams2022

    manifest = Gwilliams2022.get_dataset_manifest()
    subjects = sorted(
        {p.split("/")[0] for p in manifest if re.match(r"^sub-\d+/", p)}
    )
    assert subjects == [f"sub-{i:02d}" for i in range(1, 28)]
    # spot-check at least one events file is reachable
    sample_path = "sub-01/ses-0/meg/sub-01_ses-0_task-0_events.tsv"
    assert sample_path in manifest
