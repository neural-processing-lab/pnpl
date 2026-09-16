"""Digit-classification task for MegNIST."""

from __future__ import annotations

from ..base import BaseTask


DIGIT_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


class DigitClassification(BaseTask):
    """Classify imagined digits zero through nine."""

    _classes = DIGIT_NAMES

    def collect_samples(self, dataset) -> list[tuple]:
        """Return one sample for every epoched trial."""
        return [
            (trial_idx, int(label))
            for trial_idx, label in enumerate(dataset.labels)
        ]

    def get_label(self, sample: tuple) -> int:
        """Return the digit label for a sample."""
        return int(sample[1])