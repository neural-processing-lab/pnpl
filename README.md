# 🍍PNPL Brain Data Deep Learning Library

> The current primary use of the PNPL library is for the LibriBrain competition. [Click here](https://neural-processing-lab.github.io/2025-libribrain-competition/) to learn more and get started!

Welcome to PNPL — a Python toolkit for loading and processing brain datasets for deep learning. It provides ready‑to‑use dataset classes (PyTorch `Dataset`) and utilities with a simple, consistent API.

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

This will also take care of all requirements.

## Usage
The core functionality of the library is contained in the two Dataset classes `LibriBrainSpeech` and `LibriBrainPhoneme`.
Check out the basic usage:

### LibriBrainSpeech
This wraps the LibriBrain dataset for use in speech detection problems.
```python
from pnpl.datasets import LibriBrainSpeech

speech_example_data = LibriBrainSpeech(
    data_path="./data/",
    include_run_keys = [("0","1","Sherlock1","1")]
)

sample_data, label = speech_example_data[0]

# Print out some basic info about the sample
print("Sample data shape:", sample_data.shape)
print("Label shape:", label.shape)
```

### LibriBrainSpeech
This wraps the LibriBrain dataset for use in phoneme classification problems.
```python
from pnpl.datasets import LibriBrainPhoneme

phoneme_example_data = LibriBrainPhoneme(
    data_path="./data/",
    include_run_keys = [("0","1","Sherlock1","1")]
)
sample_data, label = phoneme_example_data[0]

# Print out some basic info about the sample
print("Sample data shape:", sample_data.shape)
print("Label shape:", label.shape)
```

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
git clone https://github.com/neural-processing-lab/pnpl-public.git
cd pnpl-public
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
