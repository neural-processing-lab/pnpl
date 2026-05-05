---
title: Gwilliams 2022 (MEG-MASC)
---

# Gwilliams 2022 (MEG-MASC)

`pnpl.datasets.Gwilliams2022` wraps the MEG-MASC corpus (Gwilliams
et al., 2022; [arXiv:2208.11488](https://arxiv.org/abs/2208.11488))
released on OSF (project [`ag3kj`](https://osf.io/ag3kj/)).

- 27 subjects, up to 2 sessions, 4 stories per session
- 208-channel KIT/Yokogawa MEG, raw `.con` files
- Open access — no auth required

The dataset is split across four sibling OSF components because of OSF's
per-component storage cap; the loader transparently aggregates files
across all four.

## Quickstart

```python
from pnpl.datasets import Gwilliams2022
from pnpl.tasks.gwilliams2022 import PhonemeClassification

ds = Gwilliams2022(
    data_path="./data/meg_masc",
    task=PhonemeClassification(tmin=-0.2, tmax=0.6),
    include_subjects=["01"],
    include_sessions=["0"],
    include_tasks=["0"],            # story 0 = "lw1"
    preprocessing="notch+bp+ds",    # 50/100 Hz notch, 0.1–125 Hz bp, 250 Hz resample
    download=True,
    standardize=True,
)

x, y = ds[0]
print(x.shape, y.item())
```

The first construction downloads the raw KIT recording (plus the
`markers.mrk` and `acq-{ELP,HSP}_headshape.pos` sidecars KIT requires),
runs the preprocessing pipeline, and caches the result as H5. Each
subject-session is roughly 3–4 GB of raw `.con` plus a few MB of
sidecars — start small, then scale up.

## BIDS axes

| Axis | Values |
| --- | --- |
| `subject` | `"01"`–`"27"` |
| `session` | `"0"`, `"1"` |
| `task` | `"0"`–`"3"` (one per story; see `TASK_STORIES`) |
| `run` | always `"01"` (MEG-MASC has no run dimension) |

Run keys are 4-tuples `(subject, session, task, run)`. The release is
not fully balanced — some subjects only have one session — so the
loader skips run keys whose files aren't in the OSF manifest and
prints a warning.

## Selected arguments

- `task` — any object implementing `TaskProtocol`. Built-ins live in
  `pnpl.tasks.gwilliams2022` (`PhonemeClassification`, `WordDetection`).
- `preprocessing` — pipeline string used in derivative filenames
  (e.g. `"notch+bp+ds"`). Set to `None` to materialize H5 from the raw
  KIT recording without filtering.
- `preprocessing_config` — per-step overrides forwarded to
  `pnpl.preprocessing.Pipeline`. See [Preprocessing](preprocessing.md).
- `include_subjects`, `include_sessions`, `include_tasks`,
  `include_run_keys` (and their `exclude_*` counterparts) — narrow the
  set of run keys.
- `standardize`, `clipping_boundary`, `channel_means`, `channel_stds`
  — see `StandardizationMixin`.
- `create_h5_if_missing` (default `True`) — when there is no cached H5
  on OSF, run the pipeline locally and cache it.

## Available tasks

```python
from pnpl.tasks.gwilliams2022 import PhonemeClassification, WordDetection
```

- `PhonemeClassification(tmin, tmax, label_type="phoneme" | "voicing")`
  — sample windowed around each phoneme onset.
- `WordDetection(tmin, tmax, require_pronounced=True)` — sample
  windowed around each word onset; label is the word string.
