"""
Word Detection task for the Pallier 2025 LittlePrince Listen dataset.

The events.tsv schema is one row per word onset:

    onset   duration   trial_type   stimulus
    7.342   0.184      Word         lorsque
    7.526   0.062      Word         j

``trial_type`` is always ``Word``; ``stimulus`` is the spoken French
token (lower-cased after stripping). We sample windows aligned to
``onset`` and label them by token. Defaults match d'Ascoli et al.,
Nat Commun 16:10521 (2025): a 3-second window starting at word onset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class WordDetection:
    """
    Word-onset classification on Pallier 2025 (LittlePrince Listen).

    Sample tuples follow the continuous-data convention:
    ``(subject, session, task, run, onset, word_str)``. The label
    vocabulary is the set of unique words observed across the requested
    runs (lower-cased, stripped).

    Args:
        tmin: Start time relative to word onset (seconds). Default 0.0.
        tmax: End time relative to word onset (seconds). Default 3.0.
        min_word_length: Minimum stripped word length to include
            (default 1; set higher to drop function words / single
            letters from the elided audiobook tokenization).
        max_word_length: Maximum stripped word length to include
            (``None`` for no limit).
        keep_top_k: If set, restrict the label vocabulary to the
            ``k`` most-frequent tokens across the requested runs.
            Other windows are dropped. Useful for the d'Ascoli et al.
            "top-250" evaluation.
    """

    tmin: float = 0.0
    tmax: float = 3.0
    min_word_length: int = 1
    max_word_length: Optional[int] = None
    keep_top_k: Optional[int] = None

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
        all_samples: list[tuple] = []
        for run_key in dataset.run_keys:
            all_samples.extend(self._collect_run(dataset, run_key))

        if self.keep_top_k is not None and self.keep_top_k > 0:
            counts: dict[str, int] = {}
            for s in all_samples:
                counts[s[5]] = counts.get(s[5], 0) + 1
            top = {
                w for w, _ in sorted(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[: self.keep_top_k]
            }
            all_samples = [s for s in all_samples if s[5] in top]

        words = sorted({s[5] for s in all_samples})
        self._words_sorted = words
        self._word_to_id = {w: i for i, w in enumerate(words)}
        return all_samples

    def _collect_run(self, dataset, run_key: tuple) -> list[tuple]:
        subject, session, task, run = run_key
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []

        if "onset" not in df.columns:
            return []

        samples: list[tuple] = []
        for _, row in df.iterrows():
            trial_type = str(row.get("trial_type", "")).strip().lower()
            if trial_type and trial_type != "word":
                continue

            stimulus = row.get("stimulus")
            if stimulus is None:
                continue
            word = str(stimulus).strip().lower()
            if not word:
                continue
            if len(word) < self.min_word_length:
                continue
            if self.max_word_length is not None and len(word) > self.max_word_length:
                continue

            try:
                onset = float(row["onset"])
            except (TypeError, ValueError):
                continue

            samples.append((subject, session, task, run, onset, word))
        return samples

    def get_label(self, sample: tuple) -> int:
        return self._word_to_id.get(sample[5], 0)
