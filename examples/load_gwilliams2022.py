"""
Load MEG-MASC (Gwilliams et al., 2022) — minimal end-to-end example.

The first construction will download the requested raw KIT recordings
(plus marker / head-shape sidecars) from OSF, run a default preprocessing
pipeline, and cache the result as H5. Subsequent constructions skip the
download/preprocess and read the cached H5 directly.

OSF storage usage is meaningful — a full subject-session is ~3-4 GB of
raw .con plus a few MB of sidecars. Start small (one subject, one
session, one task) like below, then scale up.

Run from a fresh machine:
    python examples/load_gwilliams2022.py
"""

from __future__ import annotations

from pnpl.datasets.gwilliams2022 import Gwilliams2022
from pnpl.tasks.gwilliams2022 import PhonemeClassification


def main():
    task = PhonemeClassification(tmin=-0.2, tmax=0.6, label_type="phoneme")

    dataset = Gwilliams2022(
        data_path="./data/meg_masc",
        task=task,
        include_subjects=["01"],
        include_sessions=["0"],
        include_tasks=["0"],          # story 0 = "lw1"
        preprocessing="notch+bp+ds",  # 50/100 Hz notch, 0.1-125 Hz bp, 250 Hz resample
        download=True,
        create_h5_if_missing=True,
        standardize=True,
    )

    print(f"Samples: {len(dataset)}")
    print(f"Channels: {dataset.n_channels}, time points/sample: {dataset.n_times}")
    print(f"Phoneme classes: {dataset.label_info['n_classes']}")

    x, y = dataset[0]
    print(f"First sample — x.shape={tuple(x.shape)}, label={int(y)}")


if __name__ == "__main__":
    main()
