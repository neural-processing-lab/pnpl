---
title: Datasets
---

# Public Datasets

The public `pnpl.datasets` package provides the LibriBrain 2025 dataset family plus shared utilities for deep learning workflows.

The main entry point is the task-based `LibriBrain` dataset:

```python
from pnpl.datasets import LibriBrain
from pnpl.tasks import SpeechDetection, PhonemeClassification, WordDetection
```

Additional wrapper datasets are also available:

```python
from pnpl.datasets import (
    GroupedDataset,
    LibriBrainSpeech,
    LibriBrainPhoneme,
    LibriBrainWord,
    LibriBrainSentence,
)
```

## GroupedDataset

Utility dataset to group multiple datasets and expose a unified interface.

## HDF5Dataset (base)

`pnpl.datasets.hdf5.HDF5Dataset` is a simple base for datasets backed by MEG signals serialized as HDF5, with standardization and slicing support.

Key features:
- windowed access `(channels, time)`
- channel-wise standardization
- optional clipping

## LibriBrain 2025

- `LibriBrain`: task-based dataset entry point
- `LibriBrainSpeech`: speech/silence time-series labels over a window
- `LibriBrainPhoneme`: phoneme classification from MEG segments
- `LibriBrainWord`: word-detection wrapper
- `LibriBrainSentence`: sentence-level dataset wrapper
