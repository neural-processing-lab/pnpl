---
title: LibriBrain100
---

# LibriBrain100

`pnpl.datasets.LibriBrain100` is the unified loader for the full
LibriBrain release. Conceptually it is the union of the original
[`pnpl/LibriBrain`](https://huggingface.co/datasets/pnpl/LibriBrain)
Hugging Face dataset and the extension repository
[`pnpl/LibriBrain2`](https://huggingface.co/datasets/pnpl/LibriBrain2);
there is no separate `LibriBrain100` repo, and the library hides the
split from normal users — pass a record's identifier and the loader
fetches it from whichever underlying repo owns it.

The release is structured around a **deep / broad** distinction:

- **Subject 0 (deep, ~80h)**: the entire Sherlock Holmes canon
  (`Sherlock1`..`Sherlock9`) plus TIMIT, MOCHA-TIMIT, and 30 'The
  Moth' podcast stories.
- **Subjects 1–32 (broad, ~44 min each)**: Sherlock1 ses-11 and ses-12
  only — the same chapters that have always been the validation /
  test split for the deep subject.

All recordings come from a MEGIN TRIUX™ Neo system (306 sensors,
102 magnetometers + 204 gradiometers); the serialised H5 files are
downsampled to 250 Hz.

## Quickstart

```python
from pnpl.datasets import LibriBrain100
from pnpl.tasks import SpeechDetection

ds = LibriBrain100(
    data_path="./data/LibriBrain100",
    task=SpeechDetection(tmin=0.0, tmax=0.5),
    partition="train",
)

x, y = ds[0]   # x: (channels, time), y: per-time-step speech label
```

The first construction downloads any missing files from Hugging Face;
subsequent calls read the local cache.

## Selectors

Two new selectors compose with the existing `partition` /
`include_run_keys` / `exclude_*` arguments:

```python
LibriBrain100(
    data_path=...,
    task=...,
    partition="train"|"validation"|"test"|None,
    subjects="all"|"deep"|"broad"|0|"sub-0"|[1,2,3]|range(1, 33),
    corpus="all"|"sherlock"|"timit"|"mocha"|"podcasts"|[...],
)
```

`subjects=` accepts:

| Form | Meaning |
| --- | --- |
| `"all"` (default) | every subject (sub-0 + sub-1..32) |
| `"deep"` | sub-0 only |
| `"broad"` | sub-1..32 only |
| `0`, `"0"`, `"sub-0"` | a single subject |
| `[1, 2, 3]`, `range(1, 5)` | a list of subjects |

`corpus=` accepts `"all"` (default), `"sherlock"`, `"timit"`,
`"mocha"`, `"podcasts"`, or a list of those. Aliases like
`"mocha-timit"`, `"mochatimit"`, `"the_moth"`, `"moth"` are accepted
for convenience.

## Validation rules raised early

The loader rejects two combinations up front rather than silently
returning an empty dataset:

- `subjects="broad" + corpus="timit"` (or any non-Sherlock corpus) —
  multi-subject data was only collected with the Sherlock stimuli.
- `subjects="broad" + partition="train"` — no train partition is
  assigned to the broad component by default. For SFT workflows on
  the broad subjects, use `partition="validation"` as your fine-tuning
  training set and `partition="test"` for evaluation.

## Common usage

```python
# Subject 0 only — the deep component (~80 hours):
ds = LibriBrain100(data_path=..., task=..., partition="train",
                   subjects="deep")

# Only TIMIT, only on subject 0:
ds = LibriBrain100(data_path=..., task=...,
                   subjects="deep", corpus="timit")

# Multiple corpora at once, still only on subject 0:
ds = LibriBrain100(data_path=..., task=...,
                   subjects="deep", corpus=["timit", "mocha"])
```

## Per-task wrappers

If you do not want to instantiate a task object directly:

```python
from pnpl.datasets import (
    LibriBrain100Speech,
    LibriBrain100Phoneme,
    LibriBrain100Word,
)

speech_ds = LibriBrain100Speech(data_path="./data/LibriBrain100",
                                partition="train", subjects="deep",
                                corpus="sherlock")

phoneme_ds = LibriBrain100Phoneme(data_path="./data/LibriBrain100",
                                  partition="validation",
                                  subjects="deep", corpus="sherlock",
                                  tmin=-0.2, tmax=0.6)

word_ds = LibriBrain100Word(data_path="./data/LibriBrain100",
                            include_run_keys=[("0", "1", "TheMoth", "1")])
```

The wrappers take the same `subjects` / `corpus` arguments as
`LibriBrain100`.

## Notes on partial uploads

LibriBrain2 is uploaded incrementally. While files are still landing,
records that aren't yet downloadable are skipped with a single grouped
warning at construction time, so the loader stays usable as the upload
progresses. If the warning shows up where you don't expect it, run

```python
from pnpl.datasets.libribrain100 import RUN_RECORDS
print(len(RUN_RECORDS), "records in manifest")
```

to confirm what the manifest expects.
