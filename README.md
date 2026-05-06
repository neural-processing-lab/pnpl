# 🍍PNPL Brain Data Deep Learning Library

> The current primary use of the PNPL library is for the LibriBrain competition. [Click here](https://neural-processing-lab.github.io/2025-libribrain-competition/) to learn more and get started!

Welcome to PNPL — a Python toolkit for loading and processing brain
datasets for deep learning. The package now ships four MEG dataset
loaders (LibriBrain, MEG-MASC, Armeni 2022, MOUS) plus a composable
preprocessing pipeline and shared task abstractions.

## Features
- Friendly dataset APIs backed by real MEG recordings
- Composable preprocessing pipeline (`bads+headpos+sss+notch+bp+ds`, etc.)
- On-demand download from Hugging Face (LibriBrain), OSF (MEG-MASC), Radboud WebDAV (Armeni, MOUS), and OpenNeuro (LittlePrince)
- Task-based API: pick a task object, get `(x, y)` (or `(x, y, info)`) windows
- Works with PyTorch `DataLoader` out of the box
- Clean namespace and lazy imports to keep startup fast

## Installation
```
pip install pnpl
```

This installs the package and its core dependencies.

## Usage
A common entry point uses a task object:

```python
from pnpl.datasets import LibriBrain
from pnpl.tasks import SpeechDetection

dataset = LibriBrain(
    data_path="./data/LibriBrain",
    task=SpeechDetection(tmin=0.0, tmax=0.5),
    partition="train",
)

sample_data, label = dataset[0]
print(sample_data.shape, label.shape)
```

Dataset-specific wrapper classes are also available:

```python
from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme

speech_ds = LibriBrainSpeech(data_path="./data/LibriBrain", partition="train")
phoneme_ds = LibriBrainPhoneme(data_path="./data/LibriBrain", partition="train")
```

The same task-based pattern works for the other corpora:

```python
from pnpl.datasets import Gwilliams2022, Armeni2022, Schoffelen2019
from pnpl.tasks.gwilliams2022 import PhonemeClassification

meg_masc = Gwilliams2022(
    data_path="./data/meg_masc",
    task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    include_subjects=["01"], include_sessions=["0"], include_tasks=["0"],
    preprocessing="notch+bp+ds",
)
```

For the full LibriBrain release (deep sub-0 across 9 Sherlock books +
TIMIT + MOCHA-TIMIT + 30 Moth podcasts, plus 32 broad subjects on
Sherlock1 ses-11/ses-12), use `LibriBrain100`:

```python
from pnpl.datasets import LibriBrain100
from pnpl.tasks import SpeechDetection

ds = LibriBrain100(
    data_path="./data/LibriBrain100",
    task=SpeechDetection(tmin=0.0, tmax=0.5),
    partition="train",
    subjects="deep",       # or "broad", "all", 0, [1, 2, 3], range(1, 33)
    corpus="sherlock",     # or "timit", "mocha", "podcasts", "all"
)
```

## Included Datasets

| Class | Source | Auth |
| --- | --- | --- |
| `LibriBrain` (+ `LibriBrainSpeech`/`Phoneme`/`Word`/`Sentence`) | Hugging Face `pnpl/LibriBrain` | none |
| `LibriBrain100` (+ `LibriBrain100Speech`/`Phoneme`/`Word`) | HF `pnpl/LibriBrain` ∪ `pnpl/LibriBrain2` (deep + broad release) | none |
| `Gwilliams2022` (MEG-MASC) | OSF `ag3kj` | none |
| `Armeni2022` | Radboud `DSC_3011085.05_995_v1` | Radboud credentials |
| `Schoffelen2019` (MOUS) | Radboud `DSC_3011020.09_236_v1` | Radboud credentials |
| `Pallier2025` (LittlePrince Listen) | OpenNeuro `ds007523` | none |

For the Radboud-hosted datasets, set `RADBOUD_USERNAME` and
`RADBOUD_PASSWORD` (an approved data-sharing agreement is required
before access is granted).

## Support
In case of any questions or problems, please get in touch through [our Discord server](https://discord.gg/Fqr8gJnvSh).

## Quickstart

Load a single run of the LibriBrain Speech dataset and iterate samples:

```python
from pnpl.datasets.libribrain2025 import constants
from pnpl.datasets import LibriBrainSpeech

ds = LibriBrainSpeech(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=[constants.RUN_KEYS[0]],  # pick a single run
    tmin=0.0,
    tmax=0.2,
    standardize=True,
    include_info=True,
)

print(len(ds), "samples")
x, y, info = ds[0]
print(x.shape, y.shape, info["dataset"])  # (channels,time), (time,), "libribrain2025"
```

## Documentation

We publish documentation with Jupyter Book and GitHub Pages.

- Local preview: `pip install -r docs/requirements.txt && jupyter-book build docs/` then open `docs/_build/html/index.html`.
- GitHub Pages: when made public, enable Pages via repo settings to publish automatically from the existing workflow.

The docs cover:

- Per-dataset pages (`docs/libribrain.md`, `docs/gwilliams2022.md`,
  `docs/armeni2022.md`, `docs/schoffelen2019.md`)
- The preprocessing pipeline (`docs/preprocessing.md`) and tasks
  (`docs/tasks.md`)
- Tutorials for the LibriBrain competition tracks

## Contributing
We welcome contributions from the community!

- Read the Contributor Guide in `docs/contributing.md` for setup, coding style, and PR workflow.
- Open issues for bugs and enhancements with clear, minimal repros when possible.
- Tests: add/update `pytest` tests for any feature or fix.

Quick dev setup:
```bash
git clone https://github.com/neural-processing-lab/pnpl.git
cd pnpl
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
pytest -q
```

## Support and Questions
- Check the FAQ at `docs/faq.md`.
- If something is unclear in the docs, please open a documentation issue.

## License
BSD‑3‑Clause. See `LICENSE` for details.
