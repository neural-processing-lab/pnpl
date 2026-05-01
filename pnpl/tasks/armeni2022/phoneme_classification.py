"""
Phoneme Classification task for the Armeni et al. (2022) dataset.

The Armeni events.tsv uses one row per phoneme/word onset. Phoneme rows
have ``type`` starting with ``"phoneme_onset"`` (the suffix counts the
phoneme position within the word, e.g. ``phoneme_onset_01``). Phoneme
labels are ARPABET symbols with optional stress digits (``AH0``,
``IY1``, ``EH2``); we strip the digit before mapping to a class id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ...datasets.constants import ARPABET, ARPABET_VOICELESS


@dataclass
class PhonemeClassification:
    """
    Multi-class phoneme classification on Armeni 2022.

    Args:
        tmin: Start time relative to phoneme onset (seconds).
        tmax: End time relative to phoneme onset (seconds).
        label_type: ``"phoneme"`` for multi-class, ``"voicing"`` for
            binary voiced/unvoiced.
        exclude_phonemes: ARPABET symbols to drop.
        skip_negative_onset: If True (default), drop events whose
            onset is <= 0 — pre-stimulus / sync triggers in the
            recording.
    """

    tmin: float = -0.2
    tmax: float = 0.6
    label_type: str = "phoneme"
    exclude_phonemes: list = field(default_factory=list)
    skip_negative_onset: bool = True

    _phonemes_sorted: list = field(default_factory=list, repr=False)
    _phoneme_to_id: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.label_type not in ("phoneme", "voicing"):
            raise ValueError(
                f"label_type must be 'phoneme' or 'voicing', got {self.label_type!r}"
            )

    @property
    def label_info(self) -> dict:
        if self.label_type == "voicing":
            return {
                "classes": ["uv", "v"],
                "label_to_id": {"uv": 0, "v": 1},
                "id_to_label": {0: "uv", 1: "v"},
                "n_classes": 2,
            }
        return {
            "classes": self._phonemes_sorted,
            "label_to_id": self._phoneme_to_id,
            "id_to_label": {i: p for p, i in self._phoneme_to_id.items()},
            "n_classes": len(self._phonemes_sorted),
        }

    def collect_samples(self, dataset) -> list[tuple]:
        exclude = set(self.exclude_phonemes)
        allowed = [p for p in ARPABET if p not in exclude]
        self._phonemes_sorted = allowed
        self._phoneme_to_id = {p: i for i, p in enumerate(allowed)}

        samples: list[tuple] = []
        for run_key in dataset.run_keys:
            samples.extend(self._collect_run(dataset, run_key, exclude=exclude))
        return samples

    def _collect_run(self, dataset, run_key: tuple, exclude: set[str]) -> list[tuple]:
        subject, session, task, run = run_key
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []
        if "type" not in df.columns or "value" not in df.columns:
            return []

        # Phoneme rows. ``type`` starts with ``phoneme_onset`` (e.g.
        # ``phoneme_onset_01``).
        phoneme_mask = df["type"].astype(str).str.startswith("phoneme_onset")
        df = df.loc[phoneme_mask, ["onset", "value"]].copy()
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        df = df.dropna(subset=["onset", "value"])
        if self.skip_negative_onset:
            df = df[df["onset"] > 0]

        samples: list[tuple] = []
        for _, row in df.iterrows():
            arpa = _normalize_arpabet(str(row["value"]))
            if arpa is None or arpa in exclude:
                continue
            samples.append((subject, session, task, run, float(row["onset"]), arpa))
        return samples

    def get_label(self, sample: tuple) -> int:
        arpa = sample[5]
        if self.label_type == "voicing":
            return 0 if arpa in _VOICELESS_SET else 1
        return self._phoneme_to_id.get(arpa, 0)


_VOICELESS_SET = set(ARPABET_VOICELESS)
_ARPABET_SET = set(ARPABET)


def _normalize_arpabet(value: str) -> Optional[str]:
    """Strip surrounding quotes and trailing stress digit; return the
    canonical ARPABET symbol or None if not recognized."""
    if not isinstance(value, str):
        return None
    s = value.strip().strip('"').strip("'")
    if not s:
        return None
    if s in _ARPABET_SET:
        return s
    # Trailing stress digit e.g. AH0, IY1, EH2 → strip
    if len(s) >= 2 and s[-1].isdigit() and s[:-1] in _ARPABET_SET:
        return s[:-1]
    return None
