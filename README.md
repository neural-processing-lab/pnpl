# 🍍PNPL Brain Data Deep Learning Library

> The current primary use of the PNPL library is for the LibriBrain competition. [Click here](https://neural-processing-lab.github.io/2025-libribrain-competition/) to learn more and get started!

Welcome to PNPL — a Python toolkit for loading and processing brain datasets for deep learning. The package ships the LibriBrain 2025 dataset family plus shared preprocessing and task utilities.

## Features
- Friendly dataset APIs backed by real MEG recordings
- Batteries‑included standardization, clipping, and windowing
- LibriBrain 2025 dataset support with optional on‑demand download
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

## Included Datasets
- `pnpl` includes the `libribrain2025` dataset family together with shared preprocessing and task utilities.

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
