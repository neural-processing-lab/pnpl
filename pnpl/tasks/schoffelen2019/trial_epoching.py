"""
Trial-onset epoching task for the Schöffelen et al. (2019) MOUS dataset.

The MOUS events.tsv schema is trigger-driven, not phoneme-aligned. The
canonical "trial" boundaries are rows whose ``type`` is ``"trial"``
(default 10-second blocks). Within each trial there are downstream
trigger rows (``UPPT001``, ``Picture``, ``Sound``) whose ``value`` is
either a numeric trigger code or stimulus metadata. This task pairs
each trial onset with its first ``UPPT001`` trigger as a coarse label
for sentence/listening conditions — a sensible default while leaving
the full event schema available via ``include_info``.

Pass ``label_type="binary"`` to fall back to a simpler "trial-vs-rest"
labelling (always 1 — useful as a placeholder for self-supervised
windowing without conditional labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class TrialEpoching:
    """Trial-onset epoching with a UPPT001 trigger-code label.

    Args:
        tmin: Start time relative to trial onset (seconds).
        tmax: End time relative to trial onset (seconds).
        label_type: ``"trigger"`` to label each trial with its first
            ``UPPT001`` code, or ``"binary"`` for a constant label of 1.
        include_tasks: Restrict to a subset of MOUS tasks
            (``"auditory"``, ``"visual"``, ``"rest"``); ``None`` keeps
            whatever the dataset provides.
    """

    tmin: float = 0.0
    tmax: float = 1.0
    label_type: str = "trigger"
    include_tasks: Optional[list] = None

    _classes_sorted: list = field(default_factory=list, repr=False)
    _label_to_id: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.label_type not in ("trigger", "binary"):
            raise ValueError(
                f"label_type must be 'trigger' or 'binary', got {self.label_type!r}"
            )

    @property
    def label_info(self) -> dict:
        if self.label_type == "binary":
            return {
                "classes": ["trial"],
                "label_to_id": {"trial": 1},
                "id_to_label": {1: "trial"},
                "n_classes": 1,
            }
        return {
            "classes": self._classes_sorted,
            "label_to_id": self._label_to_id,
            "id_to_label": {i: c for c, i in self._label_to_id.items()},
            "n_classes": len(self._classes_sorted),
        }

    def collect_samples(self, dataset) -> list[tuple]:
        samples_with_label: list[tuple] = []
        triggers_seen: set[str] = set()

        for run_key in dataset.run_keys:
            subject, session, task, run = run_key
            if self.include_tasks is not None and task not in self.include_tasks:
                continue
            run_samples = self._collect_run(dataset, run_key)
            for s in run_samples:
                triggers_seen.add(s[5])
            samples_with_label.extend(run_samples)

        # Stable label index built after the full pass so it doesn't
        # depend on traversal order.
        self._classes_sorted = sorted(triggers_seen)
        self._label_to_id = {c: i for i, c in enumerate(self._classes_sorted)}
        return samples_with_label

    def _collect_run(self, dataset, run_key: tuple) -> list[tuple]:
        subject, session, task, run = run_key
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []
        if "type" not in df.columns or "onset" not in df.columns:
            return []

        df = df.copy()
        df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
        df = df.dropna(subset=["onset", "type"])

        trial_mask = df["type"].astype(str).str.lower() == "trial"
        trial_rows = df.loc[trial_mask, ["onset"]].sort_values("onset")
        if trial_rows.empty:
            return []

        # Trigger lookup: for each trial onset, take the first UPPT001
        # row whose onset is >= the trial's onset.
        trig_rows = df[df["type"].astype(str).str.upper() == "UPPT001"]
        trig_rows = trig_rows.sort_values("onset")

        samples: list[tuple] = []
        for _, trial in trial_rows.iterrows():
            onset = float(trial["onset"])
            label_str = self._first_trigger_after(trig_rows, onset)
            samples.append((subject, session, task, run, onset, label_str))
        return samples

    @staticmethod
    def _first_trigger_after(trig_rows: pd.DataFrame, onset: float) -> str:
        candidate = trig_rows[trig_rows["onset"] >= onset]
        if candidate.empty:
            return "trigger:none"
        first = candidate.iloc[0]
        value = first.get("value", None)
        if pd.isna(value):
            return "trigger:none"
        # pandas reads numeric-looking trigger codes mixed with ``n/a``
        # rows as floats (10 → 10.0). Render whole-valued floats as ints
        # so labels stay stable (``trigger:10``).
        if isinstance(value, float) and float(value).is_integer():
            return f"trigger:{int(value)}"
        return f"trigger:{str(value).strip()}"

    def get_label(self, sample: tuple) -> int:
        if self.label_type == "binary":
            return 1
        return self._label_to_id.get(sample[5], 0)
