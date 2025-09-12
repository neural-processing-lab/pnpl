---
title: Datasets
---

# Public Datasets

The `pnpl.datasets` package provides dataset classes designed for deep learning workflows (PyTorch `Dataset`).

## GroupedDataset

Utility dataset to group multiple datasets and expose a unified interface.

```python
from pnpl.datasets import GroupedDataset, LibriBrainSpeech, LibriBrainPhoneme
```

## HDF5Dataset (base)

`pnpl.datasets.hdf5.HDF5Dataset` is a simple base for datasets backed by MEG signals serialized as HDF5, with standardization and slicing support.

Key features:
- windowed access `(channels, time)`
- channel-wise standardization
- optional clipping

## LibriBrain 2025

- `LibriBrainPhoneme`: phoneme classification from MEG segments.
- `LibriBrainSpeech`: speech/silence time-series labels over a window.

Both rely on a BIDS-like directory structure and can download needed files from Hugging Face.

