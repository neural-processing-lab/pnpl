---
title: Pallier 2025 (LittlePrince Listen)
---

# Pallier 2025 (LittlePrince Listen)

`pnpl.datasets.Pallier2025` wraps the audiobook-listening MEG dataset
released by Pallier et al. (2025) on OpenNeuro
([`ds007523`](https://openneuro.org/datasets/ds007523/versions/1.0.1)
— *LittlePrince_MEG_French_Listen_Pallier2025*).

- 58 native French adults × 1 session × 9 runs (one ~10-min audiobook segment per run)
- Elekta Neuromag TRIUX MEG, 306 channels (102 magnetometers + 204 planar gradiometers), raw `.fif`
- Sampled at 1000 Hz (online HP 0.1 Hz / LP 330 Hz, 50 Hz line frequency)
- Open access — no auth required
- ~478 GB total / ~8 GB per subject

The companion paper is d'Ascoli, Bel, Rapin, King et al.,
*Nat Commun* **16**:10521 (2025) —
[A zero-shot decoder of words from M/EEG signals](https://www.nature.com/articles/s41467-025-65499-0).

## Quickstart

```python
from pnpl.datasets import Pallier2025
from pnpl.tasks.pallier2025 import WordDetection

ds = Pallier2025(
    data_path="./data/pallier2025",
    task=WordDetection(tmin=0.0, tmax=3.0),
    include_subjects=["01"],
    include_runs=["01"],            # one ~10 min audiobook segment
    preprocessing="notch+bp+ds",    # 50/100 Hz notch, 0.1–125 Hz bp, 250 Hz resample
    download=True,
    standardize=True,
)

x, y = ds[0]
print(x.shape, y.item())   # (306, 750), some int word id
```

The first construction downloads the requested raw FIF (~870 MB per
run) and its `events.tsv` sidecar from OpenNeuro's public S3 bucket,
runs the preprocessing pipeline chunk-by-chunk to keep memory bounded,
and caches the result as H5. Subsequent constructions read directly
from the cached H5.

```{note}
ds007523 was recorded with Elekta active shielding (MaxShield)
enabled, which sets a header tag MNE refuses by default. The loader
acknowledges the tag and passes the raw signal through to the
preprocessing pipeline; expect a one-time `RuntimeWarning` per run on
first preprocessing. The companion paper deliberately skips MaxFilter
/ SSS, so this is the intended path.
```

## BIDS axes

| Axis | Values |
| --- | --- |
| `subject` | `"01"`–`"58"` |
| `session` | `"01"` (the only one) |
| `task` | `"listen"` (the only one) |
| `run` | `"01"`–`"09"` |

Run keys are 4-tuples `(subject, session, task, run)`. The
`include_*` / `exclude_*` filters narrow the set; pass
`include_run_keys=[(subject, session, task, run), ...]` for fully
specified inclusion.

## Reproducing the d'Ascoli 2025 preprocessing

The paper uses 0.1–40 Hz bandpass and resampling to 50 Hz, with no
notch or SSS. The pipeline is configurable on a per-step basis:

```python
ds = Pallier2025(
    data_path="./data/pallier2025",
    task=WordDetection(tmin=0.0, tmax=3.0),
    include_subjects=["01"],
    preprocessing="bp+ds",
    preprocessing_config={
        "bp": {"l_freq": 0.1, "h_freq": 40.0},
        "ds": {"sfreq": 50.0},
    },
)
```

See [Preprocessing](preprocessing.md) for the full step-config
mechanism (defaults < JSON < dataset config).

## Selected arguments

- `task` — currently `pnpl.tasks.pallier2025.WordDetection`.
- `preprocessing` — pipeline string used in derivative filenames.
  Default `"notch+bp+ds"`. `None` materializes H5 from the raw FIF
  unchanged.
- `preprocessing_config` — per-step overrides forwarded to
  `pnpl.preprocessing.Pipeline`.
- `include_subjects`, `include_sessions`, `include_tasks`,
  `include_runs`, `include_run_keys` (and their `exclude_*`
  counterparts).
- `standardize`, `clipping_boundary`, `channel_means`, `channel_stds`.
- `create_h5_if_missing` (default `True`).
- `preload_h5` (default `False`).

## Available task

`WordDetection(tmin, tmax, min_word_length, max_word_length, keep_top_k)` —
windowed around each word onset.

- `tmin`, `tmax` — defaults `0.0`, `3.0` to match d'Ascoli 2025.
- `min_word_length` — drop short / single-letter tokens. The
  audiobook tokenization includes elided particles (e.g. `j`, `l`)
  which you may want to filter out.
- `keep_top_k` — restrict to the `k` most-frequent tokens across the
  requested runs. Useful for the paper's "top-250" evaluation.

Sample tuple: `(subject, session, task, run, onset, word_str)`. The
label is the word's index in the resolved vocabulary.
