---
title: Install
---

# Install

PNPL requires Python 3.10+ and installs from PyPI:

```bash
pip install pnpl
```

Core scientific dependencies include `numpy`, `pandas`, `torch`, `h5py`, `mne`, `mne_bids`, and `huggingface_hub`.

## Development install (editable)

```bash
git clone https://github.com/neural-processing-lab/pnpl.git
cd pnpl
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
