---
title: Quickstart
---

# Quickstart

This page shows short examples loading datasets and iterating samples.

## LibriBrain Speech (public)

```python
from pnpl.datasets.libribrain2025 import constants
from pnpl.datasets import LibriBrainSpeech

# pick one run to keep it quick
include_run_keys = [constants.RUN_KEYS[0]]  # e.g. ('0','1','Sherlock1','1')

ds = LibriBrainSpeech(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=include_run_keys,
    tmin=0.0,
    tmax=0.2,
    standardize=True,
    include_info=True,
)

print(len(ds), "samples")
x, y, info = ds[0]
print(x.shape, y.shape, info["dataset"])  # (channels,time), (time,), "libribrain2025"
```

## LibriBrain Phoneme (public)

```python
from pnpl.datasets.libribrain2025 import constants
from pnpl.datasets import LibriBrainPhoneme

include_run_keys = [constants.RUN_KEYS[0]]

ds = LibriBrainPhoneme(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=include_run_keys,
    tmin=-0.2,
    tmax=0.6,
    standardize=True,
)

print(len(ds), "samples")
x, y = ds[0]
print(x.shape, y.item())
```

```{note}
The first time you instantiate a dataset with `download=True` (default), required files are downloaded from Hugging Face and cached under `data_path`.
```

