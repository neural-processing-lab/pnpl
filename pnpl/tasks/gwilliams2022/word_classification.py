"""
Word Classification task for the MEG-MASC dataset (Gwilliams et al., 2022).

Each sample is aligned to a word onset; the label is the lower-cased word
string. Useful for word-onset decoding or word-frequency regression.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class WordClassification:
    """
    Word-onset classification on MEG-MASC.

    Sample tuples follow the continuous-data convention:
    ``(subject, session, task, run, onset, word_str)``. By default the
    label vocabulary is the set of unique words observed across the
    requested runs.
    """

    tmin: float = -0.2
    tmax: float = 0.6
    require_pronounced: bool = True

    _words_sorted: list = field(default_factory=list, repr=False)
    _word_to_id: dict = field(default_factory=dict, repr=False)

    @property
    def label_info(self) -> dict:
        return {
            "classes": self._words_sorted,
            "label_to_id": self._word_to_id,
            "id_to_label": {i: w for w, i in self._word_to_id.items()},
            "n_classes": len(self._words_sorted),
        }

    def collect_samples(self, dataset) -> list[tuple]:
        samples: list[tuple] = []
        for run_key in dataset.run_keys:
            samples.extend(self._collect_run(dataset, run_key))

        words = sorted({s[5] for s in samples})
        self._words_sorted = words
        self._word_to_id = {w: i for i, w in enumerate(words)}
        return samples

    def _collect_run(self, dataset, run_key: tuple) -> list[tuple]:
        subject, session, task, run = run_key
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []

        samples: list[tuple] = []
        for _, row in df.iterrows():
            meta = _parse_trial_type(row.get("trial_type"))
            if meta is None or meta.get("kind") != "word":
                continue
            if self.require_pronounced and not meta.get("pronounced", 1):
                continue

            word = str(meta.get("word", "")).strip().lower()
            if not word:
                continue

            onset = float(row["onset"])
            samples.append((subject, session, task, run, onset, word))
        return samples

    def get_label(self, sample: tuple) -> int:
        return self._word_to_id.get(sample[5], 0)


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


# Backwards-compatible alias for code written against pnpl <= 0.1.0.
WordDetection = WordClassification


__all__ = ["WordClassification", "WordDetection"]
