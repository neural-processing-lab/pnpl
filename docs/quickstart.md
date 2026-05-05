---
title: Quickstart
---

# Quickstart

Every dataset in `pnpl.datasets` follows the same shape: pass a `task`
object from `pnpl.tasks` and (optionally) a `preprocessing` string,
then iterate samples like any PyTorch `Dataset`.

## LibriBrain (Hugging Face, no auth)

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

LibriBrain has dataset-specific wrapper classes that don't require a
separate task object:

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
```

## MEG-MASC / Gwilliams 2022 (OSF, no auth)

```python
from pnpl.datasets import Gwilliams2022
from pnpl.tasks.gwilliams2022 import PhonemeClassification

ds = Gwilliams2022(
    data_path="./data/meg_masc",
    task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    include_subjects=["01"],
    include_sessions=["0"],
    include_tasks=["0"],          # story 0 = "lw1"
    preprocessing="notch+bp+ds",
    download=True,
    standardize=True,
)
x, y = ds[0]
```

## Armeni 2022 (Radboud, auth required)

```python
import os
os.environ["RADBOUD_USERNAME"] = "you@orcid.org"
os.environ["RADBOUD_PASSWORD"] = "..."

from pnpl.datasets import Armeni2022
from pnpl.tasks.armeni2022 import PhonemeClassification

ds = Armeni2022(
    data_path="./data/armeni",
    task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    include_subjects=["001"],
    include_sessions=["001"],
    preprocessing="notch+bp+ds",
    standardize=True,
)
```

## Schöffelen 2019 / MOUS (Radboud, auth required)

```python
from pnpl.datasets import Schoffelen2019
from pnpl.tasks.schoffelen2019 import TrialEpoching

ds = Schoffelen2019(
    data_path="./data/schoffelen",
    task=TrialEpoching(tmin=0.0, tmax=1.0, label_type="trigger"),
    include_subjects=["A2002"],
    include_tasks=["auditory"],
    preprocessing="notch+bp+ds",
    standardize=True,
)
```

```{note}
The first time you instantiate a non-LibriBrain dataset, files are
downloaded from the appropriate remote (OSF for MEG-MASC, Radboud
WebDAV for Armeni / MOUS) and the preprocessing pipeline runs against
the raw recording, caching the result as H5 under
`data_path/derivatives/serialised/...`. Subsequent constructions read
the cached H5 directly.
```

For LibriBrain, files are downloaded from Hugging Face and cached
under `data_path` on first use.

See [Datasets](datasets.md) for the full list of arguments and
[Preprocessing](preprocessing.md) for how to customize the pipeline.
