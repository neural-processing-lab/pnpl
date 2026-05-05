---
title: Install
---

# Install

PNPL requires Python 3.10+ and installs from PyPI:

```bash
pip install pnpl
```

Core scientific dependencies include `numpy`, `pandas`, `torch`,
`h5py`, `mne`, `mne_bids`, `huggingface_hub`, and `requests` (used by
the OSF and Radboud download backends).

## Authentication for gated datasets

The Radboud-hosted datasets (`Armeni2022`, `Schoffelen2019`) require an
approved data-sharing agreement plus credentials. Set them in your
environment (or a project-local `.env` next to your code — `pnpl`
loads `.env` automatically):

```bash
export RADBOUD_USERNAME="you@orcid.org"   # often an ORCID
export RADBOUD_PASSWORD="..."
```

LibriBrain (Hugging Face) and MEG-MASC (OSF) are open and require no
credentials.

## Development install (editable)

```bash
git clone https://github.com/neural-processing-lab/pnpl.git
cd pnpl
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
