---
title: Preprocessing
---

# Preprocessing

`pnpl.preprocessing` is a small composable pipeline for turning raw
MNE recordings into the H5 files the dataset classes consume. Each
non-LibriBrain dataset (`Gwilliams2022`, `Armeni2022`, `Schoffelen2019`)
runs this pipeline automatically on the first construction; you can
also use it directly for offline preprocessing.

## Pipelines

```python
from pnpl.preprocessing import Pipeline

# String shorthand: 50/100 Hz notch, 0.1–125 Hz bandpass, 250 Hz resample
pipeline = Pipeline.from_string("notch+bp+ds")

# Or assemble step objects yourself
from pnpl.preprocessing import NotchFilter, BandpassFilter, Downsample
pipeline = Pipeline([
    NotchFilter(freqs=[50.0, 100.0]),
    BandpassFilter(l_freq=0.1, h_freq=125.0),
    Downsample(sfreq=250.0),
])

raw = pipeline.run(
    raw,                    # mne.io.Raw
    subject="01", session="0", task="0", run="01",
    bids_root="./data/meg_masc",
)
```

A pipeline is just a list of `BaseStep` objects with a `run()` method
that threads each step over the raw data and shares context (e.g. bad
channels, head positions) between them.

## Available steps

| Short name | Class | Purpose |
| --- | --- | --- |
| `bads` | `BadChannels` | Detect noisy/flat channels via `find_bad_channels_maxwell` |
| `headpos` | `HeadPosition` | Load cached head-position CSV (used by SSS) |
| `sss` | `MaxwellFilter` | Signal Space Separation (incl. head-position correction) |
| `notch` | `NotchFilter` | Notch out line noise (default `[50, 100] Hz`) |
| `bp` | `BandpassFilter` | Bandpass filter (default 0.1–125 Hz) |
| `ds` | `Downsample` | Resample (default 250 Hz) |
| `epo` | `Epoch` | Cut continuous data into epochs around stim events |

The order in `from_string("a+b+c")` is the order steps are applied. The
LibriBrain canonical recipe is `"bads+headpos+sss+notch+bp+ds"`; the
non-LibriBrain datasets default to `"notch+bp+ds"` (no SSS — SSS is
Elekta-specific and the other datasets use KIT or CTF systems).

## Configuring step parameters

Three layers, in increasing precedence:

1. **Step defaults** — dataclass field defaults on each step class.
2. **JSON config** — `{data_path}/preprocessing_config.json`, keyed by
   step name. Useful for keeping per-dataset preprocessing tuned in
   one place without editing code.
3. **Dataset config** — pass `preprocessing_config={...}` to the
   dataset constructor. Highest priority; overrides JSON and defaults.

Example JSON file at `./data/meg_masc/preprocessing_config.json`:

```json
{
  "notch": {"freqs": [50.0, 100.0, 150.0]},
  "bp":    {"l_freq": 0.5, "h_freq": 40.0},
  "ds":    {"sfreq": 200.0}
}
```

Equivalent dataset-side override (wins over the JSON file above):

```python
from pnpl.datasets import Gwilliams2022
from pnpl.tasks.gwilliams2022 import PhonemeClassification

ds = Gwilliams2022(
    data_path="./data/meg_masc",
    task=PhonemeClassification(),
    preprocessing="notch+bp+ds",
    preprocessing_config={
        "bp": {"l_freq": 0.5, "h_freq": 40.0},
        "ds": {"sfreq": 200.0},
    },
)
```

`pnpl.preprocessing.config.resolve_preprocessing_config()` is the
function that does the merge; it also tracks where each parameter
came from for logging.

## Serialization

`pnpl.preprocessing.serialization` contains the helpers the dataset
classes use to cache preprocessed data:

- `fif_to_h5(raw, output_path)` — write a continuous `mne.io.Raw` to
  H5 with `data` (channels × time), `times`, and `sample_frequency`,
  `highpass_cutoff`, `lowpass_cutoff`, `channel_names`,
  `channel_types` attributes.
- `epochs_to_h5(epochs, output_path)` — write `mne.Epochs` to H5
  (`data` is `trials × channels × time`; includes `labels`, sensor
  positions, channel metadata).

## Writing a custom step

Any subclass of `BaseStep` decorated with `@register_step("name")`
becomes available in `Pipeline.from_string`:

```python
from dataclasses import dataclass
from pnpl.preprocessing import BaseStep
from pnpl.preprocessing.pipeline import register_step

@register_step("rectify")
@dataclass
class Rectify(BaseStep):
    step_name: str = "rectify"

    def apply(self, raw, context):
        raw.apply_function(lambda x: abs(x), picks="meg")
        return raw

Pipeline.from_string("notch+bp+ds+rectify")
```

`context` is a shared dict that propagates between steps. Existing
steps put bad channels under `context["bad_channels"]` and head
positions under `context["head_pos"]` so that `MaxwellFilter` can pick
them up.
