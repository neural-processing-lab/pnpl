---
title: Schöffelen 2019 (MOUS)
---

# Schöffelen 2019 (MOUS)

`pnpl.datasets.Schoffelen2019` wraps the "Mother of Unification
Studies" (MOUS) sentence-comprehension MEG dataset (Schöffelen et al.,
2019) released on the Radboud Data Repository
([`DSC_3011020.09_236_v1`](https://webdav.data.ru.nl/dccn/DSC_3011020.09_236_v1/)).

- ~200 participants, two cohorts:
  - subjects with the `A*` prefix performed the **auditory** task
  - subjects with the `V*` prefix performed the **visual** (reading)
    task
- every participant performed a shared **rest** block
- CTF MEG, raw `.ds` directories
- ~2 GB raw CTF per task

```{warning}
MOUS is **not** open access. You need an approved data-sharing
agreement with the dataset owner before you can download.
```

## Auth

```bash
export RADBOUD_USERNAME="you@orcid.org"
export RADBOUD_PASSWORD="..."
```

## Quickstart

```python
from pnpl.datasets import Schoffelen2019
from pnpl.tasks.schoffelen2019 import TrialEpoching

ds = Schoffelen2019(
    data_path="./data/schoffelen",
    task=TrialEpoching(tmin=0.0, tmax=1.0, label_type="trigger"),
    include_subjects=["A2002"],
    include_tasks=["auditory"],
    preprocessing="notch+bp+ds",
    download=True,
    standardize=True,
)

x, y = ds[0]
print(x.shape, y.item())
```

A good starting point is a single `A*` (or `V*`) subject with the
`rest` task — `rest` is smaller and needs no stimulus alignment.

## Layout differences from the other datasets

MOUS doesn't have a session axis. The BIDS layout is
`sub-XXXX/meg/sub-XXXX_task-{auditory,visual,rest}_meg.ds`, so the
loader synthesizes a constant `session = "01"` for the standard
`(subject, session, task, run)` 4-tuple convention. It also auto-skips
tasks the subject didn't perform (`auditory` for `V*`, `visual` for
`A*`).

## BIDS axes

| Axis | Values |
| --- | --- |
| `subject` | `A2002`–`A2125` (auditory cohort), `V1001`–`V1117` (visual cohort) |
| `session` | always `"01"` (synthesized) |
| `task` | `"auditory"`, `"visual"`, `"rest"` (gated per-subject) |
| `run` | always `"01"` |

## Selected arguments

- `task` — currently `pnpl.tasks.schoffelen2019.TrialEpoching`.
- `preprocessing`, `preprocessing_config` — see [Preprocessing](preprocessing.md).
- `include_subjects`, `include_tasks`, `include_run_keys` (and their
  `exclude_*` counterparts).
- `standardize`, `clipping_boundary`, `channel_means`, `channel_stds`.
- `create_h5_if_missing`, `preload_h5`.

## Available tasks

`TrialEpoching(tmin, tmax, label_type)` epochs around each trial onset
(rows whose `type` is `"trial"` in `events.tsv`).

- `label_type="trigger"` (default) — label each trial with the first
  `UPPT001` trigger code that appears inside it.
- `label_type="binary"` — constant label of `1` (useful for
  self-supervised windowing without conditional labels).
