"""
Phoneme Classification task for the MEG-MASC dataset (Gwilliams et al., 2022).

MEG-MASC events.tsv stores trial metadata as a Python-literal dict in the
``trial_type`` column, with three event kinds: ``phoneme``, ``word``,
``sound``. Phonemes use TIMIT notation with a position suffix
(``t_B``, ``eh_I``, ``r_E``, ``s_S``) — we map TIMIT → ARPABET and discard
silence/pause symbols, matching the convention in
``pnpl.datasets.constants``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ...datasets.constants import ARPABET, ARPABET_VOICELESS, TIMIT_TO_ARPABET


@dataclass
class PhonemeClassification:
    """
    Multi-class phoneme classification on MEG-MASC.

    Each sample is aligned to a phoneme onset extracted from the events
    file. Labels are ARPABET phonemes (after mapping from TIMIT). Sample
    tuples follow the continuous-data convention used elsewhere in pnpl:
    ``(subject, session, task, run, onset, label_str)``.

    Args:
        tmin: Start time relative to phoneme onset (seconds).
        tmax: End time relative to phoneme onset (seconds).
        label_type: ``"phoneme"`` for multi-class, ``"voicing"`` for binary
            voiced/unvoiced.
        exclude_phonemes: ARPABET symbols to drop from samples and the
            label vocabulary.
        require_pronounced: If True, drop events with ``pronounced`` set
            to a falsy value (e.g. silently-read trials).
    """

    tmin: float = -0.2
    tmax: float = 0.6
    label_type: str = "phoneme"
    exclude_phonemes: list = field(default_factory=list)
    require_pronounced: bool = True

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

        samples: list[tuple] = []
        for _, row in df.iterrows():
            meta = _parse_trial_type(row.get("trial_type"))
            if meta is None or meta.get("kind") != "phoneme":
                continue
            if self.require_pronounced and not meta.get("pronounced", 1):
                continue

            timit = str(meta.get("phoneme", "")).split("_")[0]
            if not timit:
                continue
            arpa = TIMIT_TO_ARPABET.get(timit)
            if arpa is None or arpa == "SIL" or arpa in exclude:
                continue

            onset = float(row["onset"])
            samples.append((subject, session, task, run, onset, arpa))
        return samples

    def get_label(self, sample: tuple) -> int:
        arpa = sample[5]
        if self.label_type == "voicing":
            return 0 if arpa in _VOICELESS_SET else 1
        return self._phoneme_to_id.get(arpa, 0)


_VOICELESS_SET = set(ARPABET_VOICELESS)


def _parse_trial_type(value) -> Optional[dict]:
    if not isinstance(value, str):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
