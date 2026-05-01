---
title: Quickstart
---

# Quickstart

This page shows the task-based LibriBrain entry point together with dataset-specific wrappers.

## Task-based API

```python
from pnpl.datasets import LibriBrain
from pnpl.tasks import SpeechDetection

ds = LibriBrain(
    data_path="./data/LibriBrain",
    task=SpeechDetection(tmin=0.0, tmax=0.2),
    partition="train",
    standardize=True,
    include_info=True,
)

print(len(ds), "samples")
x, y, info = ds[0]
print(x.shape, y.shape, info["dataset"])  # (channels,time), (time,), "libribrain2025"
```

## Wrapper datasets

```python
from pnpl.datasets.libribrain2025 import constants
from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme

include_run_keys = [constants.RUN_KEYS[0]]

speech_ds = LibriBrainSpeech(
    data_path="./data/LibriBrain",
    include_run_keys=include_run_keys,
    tmin=0.0,
    tmax=0.2,
)

phoneme_ds = LibriBrainPhoneme(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=include_run_keys,
    tmin=-0.2,
    tmax=0.6,
    standardize=True,
)

print(len(speech_ds), "speech samples")
print(len(phoneme_ds), "phoneme samples")
x, y = phoneme_ds[0]
print(x.shape, y.item())
```

```{note}
The first time you instantiate a dataset with `download=True` (default), required files are downloaded from Hugging Face and cached under `data_path`.
```
