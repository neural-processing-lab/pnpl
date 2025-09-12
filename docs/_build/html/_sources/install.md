---
title: Install
---

# Install

PNPL requires Python 3.10+ and installs from PyPI:

```bash
pip install pnpl
```

Core scientific dependencies include `numpy`, `pandas`, `torch`, `h5py`, `mne`, `mne_bids`, and `huggingface_hub`.

```{tip}
To use private/internal datasets as part of the same `pnpl` namespace, also install the overlay package `pnpl-internal` from your private index (or editable checkout). The overlay depends on `pnpl` and contributes additional modules under `pnpl.*`.
```

## Development install (editable)

```bash
git clone https://github.com/neural-processing-lab/pnpl-public.git
cd pnpl-public
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

