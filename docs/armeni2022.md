---
title: Armeni 2022
---

# Armeni 2022

`pnpl.datasets.Armeni2022` wraps the audiobook-listening MEG dataset
released by Armeni et al. (2022) on the Radboud Data Repository
([`DSC_3011085.05_995_v1`](https://webdav.data.ru.nl/dccn/DSC_3011085.05_995_v1/)).

- 3 subjects × 10 sessions, ~10 hours of audiobook per subject
- CTF275 axial gradiometer system, raw `.ds` directories
- Sessions are large (~8 GB raw CTF each)
- Single task: `compr` (comprehension)

```{warning}
Armeni 2022 is **not** open access. You need an approved data-sharing
agreement with the dataset owner before you can download.
```

## Auth

```bash
export RADBOUD_USERNAME="you@orcid.org"   # often an ORCID
export RADBOUD_PASSWORD="..."
```

`pnpl` reads these from the environment (or a project-local `.env`)
and uses HTTP Basic auth against the Radboud WebDAV endpoint.

## Quickstart

```python
from pnpl.datasets import Armeni2022
from pnpl.tasks.armeni2022 import PhonemeClassification

ds = Armeni2022(
    data_path="./data/armeni",
    task=PhonemeClassification(tmin=-0.2, tmax=0.6, label_type="phoneme"),
    include_subjects=["001"],
    include_sessions=["001"],
    include_tasks=["compr"],
    preprocessing="notch+bp+ds",
    download=True,
    standardize=True,
)

x, y = ds[0]
print(x.shape, y.item())
```

The first construction downloads the requested CTF `.ds` directory
(every chunked binary inside it), runs the preprocessing pipeline, and
caches the result as H5. Subsequent constructions read directly from
the cached H5.

```{note}
Because raw CTF sessions are multi-GB, the Armeni loader runs the
pipeline **chunk-by-chunk** (default 120-second chunks) so peak memory
stays bounded. Reference / EEG / EOG / STIM channels are dropped before
filtering — only MEG channels are preserved.
```

## BIDS axes

| Axis | Values |
| --- | --- |
| `subject` | `"001"`, `"002"`, `"003"` |
| `session` | `"001"`–`"010"` |
| `task` | `"compr"` |
| `run` | always `"01"` (Armeni has no run dimension) |

## Selected arguments

- `task` — currently `pnpl.tasks.armeni2022.PhonemeClassification`.
- `preprocessing` — pipeline string used in derivative filenames
  (e.g. `"notch+bp+ds"`). Set to `None` to materialize H5 unchanged
  from the raw CTF.
- `preprocessing_config` — per-step overrides forwarded to
  `pnpl.preprocessing.Pipeline`. See [Preprocessing](preprocessing.md).
- `include_subjects`, `include_sessions`, `include_tasks`,
  `include_run_keys` (and their `exclude_*` counterparts).
- `standardize`, `clipping_boundary`, `channel_means`, `channel_stds`.
- `create_h5_if_missing` (default `True`).
- `preload_h5` (default `False`) — read the H5 fully into RAM on first
  access. Faster repeat reads at the cost of memory.
