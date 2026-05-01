"""
Load Schöffelen et al. (2019) MOUS — minimal end-to-end example.

The first construction downloads the requested CTF ``.ds`` directory
(plus events sidecar) from the Radboud Data Repository, runs the
default preprocessing pipeline, and caches the result as H5.

Each task ``.ds`` is ~2 GB raw. Start with a single subject and the
``rest`` task (smaller, no stimulus alignment needed) before scaling up.

Set the env vars first:
    export RADBOUD_USERNAME="you@orcid.org"
    export RADBOUD_PASSWORD="..."

Then:
    python examples/load_schoffelen2019.py
"""

from __future__ import annotations

from pnpl.datasets.schoffelen2019 import Schoffelen2019
from pnpl.tasks.schoffelen2019 import TrialEpoching


def main():
    task = TrialEpoching(tmin=0.0, tmax=1.0, label_type="trigger")

    dataset = Schoffelen2019(
        data_path="./data/schoffelen",
        task=task,
        include_subjects=["A2002"],
        include_tasks=["auditory"],
        preprocessing="notch+bp+ds",
        download=True,
        create_h5_if_missing=True,
        standardize=True,
    )

    print(f"Samples: {len(dataset)}")
    print(f"Channels: {dataset.n_channels}, time points/sample: {dataset.n_times}")
    print(f"Trigger classes: {dataset.label_info['n_classes']}")

    x, y = dataset[0]
    print(f"First sample — x.shape={tuple(x.shape)}, label={int(y)}")


if __name__ == "__main__":
    main()
