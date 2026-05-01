"""
Load Armeni et al. (2022) — minimal end-to-end example.

The first construction downloads the requested CTF ``.ds`` directory
(and its events sidecar) from the Radboud Data Repository, runs the
default preprocessing pipeline, and caches the result as H5.

Note that Armeni recordings are large (~8 GB per session in raw CTF
form). Start small (one subject, one session) before scaling up.

Set the env vars first:
    export RADBOUD_USERNAME="you@orcid.org"
    export RADBOUD_PASSWORD="..."

Then:
    python examples/load_armeni2022.py
"""

from __future__ import annotations

from pnpl.datasets.armeni2022 import Armeni2022
from pnpl.tasks.armeni2022 import PhonemeClassification


def main():
    task = PhonemeClassification(tmin=-0.2, tmax=0.6, label_type="phoneme")

    dataset = Armeni2022(
        data_path="./data/armeni",
        task=task,
        include_subjects=["001"],
        include_sessions=["001"],
        include_tasks=["compr"],
        preprocessing="notch+bp+ds",
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
