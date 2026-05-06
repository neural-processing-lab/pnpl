"""
Load Pallier 2025 — LittlePrince Listen — minimal end-to-end example.

The first construction downloads the requested raw FIF (and its
events.tsv sidecar) from OpenNeuro ds007523, runs the default
preprocessing pipeline, and caches the result as H5. Subsequent
constructions skip the download/preprocess and read the cached H5
directly.

A single (subject, run) is ~870 MB raw FIF and ~10 minutes of audio.
The full dataset is ~478 GB. Start small.

Run from a fresh machine:
    python examples/load_pallier2025.py
"""

from __future__ import annotations

from pnpl.datasets import Pallier2025
from pnpl.tasks.pallier2025 import WordClassification


def main():
    task = WordClassification(tmin=0.0, tmax=3.0)

    dataset = Pallier2025(
        data_path="./data/pallier2025",
        task=task,
        include_subjects=["01"],
        include_runs=["01"],            # one ~10 min audiobook segment
        preprocessing="notch+bp+ds",    # 50/100 Hz notch, 0.1-125 Hz bp, 250 Hz resample
        download=True,
        create_h5_if_missing=True,
        standardize=True,
    )

    print(f"Samples: {len(dataset)}")
    print(f"Channels: {dataset.n_channels}, time points/sample: {dataset.n_times}")
    print(f"Word vocab size: {dataset.label_info['n_classes']}")

    x, y = dataset[0]
    print(f"First sample — x.shape={tuple(x.shape)}, label={int(y)}")


if __name__ == "__main__":
    main()
