---
title: Tasks
---

# Tasks

A `Task` decides how a dataset's continuous (or epoched) data is turned
into supervised samples. Pass a task instance to a dataset's `task=`
argument and the dataset will:

1. call `task.collect_samples(self)` during `__init__` to enumerate
   sample tuples (one per phoneme onset, word onset, trial onset, …),
2. call `task.get_label(sample)` inside `__getitem__` to compute the
   label, and
3. expose `task.label_info` (classes, label↔id maps, `n_classes`).

Most tasks are dataclasses with a `tmin` and `tmax` (window edges in
seconds) plus task-specific knobs. They are *cheap* — instantiate one
per dataset.

## TaskProtocol

```python
from pnpl.tasks import TaskProtocol
```

A minimal task implements three things:

```python
@dataclass
class MyTask:
    tmin: float = 0.0
    tmax: float = 0.5

    def collect_samples(self, dataset) -> list[tuple]:
        ...

    def get_label(self, sample: tuple) -> Any:
        ...

    @property
    def label_info(self) -> dict:
        return {"classes": [...], "label_to_id": {...}, "n_classes": ...}
```

`pnpl.tasks.base.BaseTask` is an optional convenience base class with a
default `label_info` implementation if you set `_classes` /
`_label_to_id`.

## LibriBrain tasks

```python
from pnpl.tasks import (
    SpeechDetection,
    PhonemeClassification,
    WordDetection,
)
```

(Re-exported at `pnpl.tasks` for convenience; the canonical module is
`pnpl.tasks.libribrain`.)

- `SpeechDetection(tmin, tmax, stride=None, oversample_silence_jitter=0)`
  — slide a window across continuous MEG and label each step as
  *speech* / *silence*. Returns a per-time-point label array.
- `PhonemeClassification(tmin, tmax, label_type="phoneme" | "voicing", exclude_phonemes=[])`
  — sample windowed around each phoneme onset.
- `WordDetection(tmin, tmax, min_word_length=1, max_word_length=None, keyword_detection=None)`
  — multi-class word classification *or* binary keyword detection
  (`keyword_detection="cat"` → 1 if the window's word is `"cat"`, else
  0). `tmin` / `tmax` may be `None` to auto-compute from word duration.

## MEG-MASC tasks (Gwilliams 2022)

```python
from pnpl.tasks.gwilliams2022 import PhonemeClassification, WordDetection
```

- `PhonemeClassification(tmin, tmax, label_type="phoneme" | "voicing")`
  — phoneme-aligned epochs from the MEG-MASC events.tsv.
- `WordDetection(tmin, tmax, require_pronounced=True)` — word-aligned
  epochs; label is the lower-cased word string.

## Armeni 2022 tasks

```python
from pnpl.tasks.armeni2022 import PhonemeClassification
```

- `PhonemeClassification(tmin, tmax, label_type, exclude_phonemes, skip_negative_onset)`
  — phoneme-aligned epochs. ARPABET stress digits (`AH0`, `IY1`, …)
  are stripped before mapping to class ids.

## Schöffelen 2019 (MOUS) tasks

```python
from pnpl.tasks.schoffelen2019 import TrialEpoching
```

- `TrialEpoching(tmin, tmax, label_type="trigger" | "binary", include_tasks=None)`
  — epochs around each trial onset. `"trigger"` labels with the first
  `UPPT001` trigger code inside the trial; `"binary"` labels every
  trial with a constant `1`.

## Sample tuple convention

For continuous-data tasks, sample tuples follow:

```
(subject, session, task, run, onset, label_value, ...)
```

`label_value` is a string for tasks like word detection, an integer
trigger code for MOUS, or a phoneme symbol for phoneme classification.
The dataset translates it to the final tensor label via
`task.get_label(sample)` and the `label_info` lookup.
