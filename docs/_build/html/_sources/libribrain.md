---
title: LibriBrain
---

# LibriBrain

The LibriBrain 2025 datasets provide MEG-based tasks with convenient download and caching from Hugging Face.

## Common Arguments

- `data_path`: local root where files are stored / downloaded
- `preprocessing_str`: expected preprocessing string in filenames
- `tmin`, `tmax`: window relative to event (seconds)
- `standardize`: z-score channels using per-run stats
- `include_run_keys`: list of run keys to include (see constants.RUN_KEYS)
- `include_info`: include an info dict in each sample
- `download`: if True (default), fetch missing files via Hugging Face

## Speech (binary time series)

```python
from pnpl.datasets import LibriBrainSpeech
from pnpl.datasets.libribrain2025 import constants

ds = LibriBrainSpeech(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=[constants.RUN_KEYS[0]],
    tmin=0.0,
    tmax=0.2,
    include_info=True,
)

print(len(ds))
```

Each item returns `(data: float32[channels,time], labels: int[time], info: dict)`.

## Phoneme (classification)

```python
from pnpl.datasets import LibriBrainPhoneme
from pnpl.datasets.libribrain2025 import constants

ds = LibriBrainPhoneme(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=[constants.RUN_KEYS[0]],
    tmin=-0.2,
    tmax=0.6,
)
print(len(ds))
```

Each item returns `(data: float32[channels,time], label_id: int64)`.

