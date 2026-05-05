---
title: Datasets
---

# Public Datasets

`pnpl.datasets` ships ready-to-use PyTorch `Dataset` classes for several
public MEG corpora. Each dataset class follows the same shape: pass a
`task` object (from `pnpl.tasks`) plus optional include/exclude filters,
and the dataset materializes preprocessed H5 files on demand.

| Class | Source | Auth | Format on disk |
| --- | --- | --- | --- |
| `LibriBrain` | Hugging Face `pnpl/LibriBrain` | none | preprocessed H5 |
| `Gwilliams2022` | OSF `ag3kj` (MEG-MASC) | none | KIT `.con` → H5 |
| `Armeni2022` | Radboud `DSC_3011085.05_995_v1` | Radboud login | CTF `.ds` → H5 |
| `Schoffelen2019` | Radboud `DSC_3011020.09_236_v1` (MOUS) | Radboud login | CTF `.ds` → H5 |

Common imports:

```python
from pnpl.datasets import (
    # Task-based entry points
    LibriBrain,
    Gwilliams2022,
    Armeni2022,
    Schoffelen2019,
    # LibriBrain wrappers (no `task=` argument needed)
    LibriBrainSpeech,
    LibriBrainPhoneme,
    LibriBrainWord,
    LibriBrainSentence,
    # Utilities
    GroupedDataset,
)
```

See the per-dataset pages for end-to-end examples:

- [LibriBrain](libribrain.md) — speech, phoneme, word, sentence tasks
- [Gwilliams 2022 (MEG-MASC)](gwilliams2022.md) — story listening, OSF
- [Armeni 2022](armeni2022.md) — audiobook listening, Radboud
- [Schöffelen 2019 (MOUS)](schoffelen2019.md) — sentence comprehension, Radboud

## Anatomy of a dataset class

Every dataset class composes the same building blocks (see
`pnpl.datasets.mixins`):

- a **download mixin** (`HFDownloadMixin`, `OSFDownloadMixin`,
  `RadboudDownloadMixin`) that knows how to fetch missing files from the
  appropriate remote
- `BIDSMixin` for BIDS-style path resolution (`sub-XXX/ses-XXX/...`)
- `ContinuousH5Mixin` for windowed reads from H5
- `StandardizationMixin` for per-channel z-scoring + outlier clipping

The first time you instantiate a dataset, missing files are downloaded
and — for the non-LibriBrain corpora — a configurable preprocessing
pipeline is run against the raw recording and cached as H5. Subsequent
constructions read directly from the cached H5.

### Auth for Radboud datasets

`Armeni2022` and `Schoffelen2019` are gated. Once your data-sharing
agreement is approved, set:

```bash
export RADBOUD_USERNAME="you@orcid.org"   # often an ORCID
export RADBOUD_PASSWORD="..."
```

`pnpl` reads these from the environment (or a local `.env` next to your
project) and uses HTTP Basic auth against the WebDAV endpoint.

## Task object

All four task-based datasets take a `task=` argument that conforms to
`pnpl.tasks.base.TaskProtocol`. A task object decides:

1. how raw events are turned into sample tuples (`collect_samples`),
2. how each sample's label is computed (`get_label`), and
3. what the label vocabulary looks like (`label_info`).

Task classes live under `pnpl.tasks.<dataset>` (LibriBrain tasks are
also re-exported at `pnpl.tasks` for convenience). See the
[Tasks](tasks.md) page for the full list.

## GroupedDataset

`GroupedDataset` wraps another dataset and groups consecutive samples
that share a label. Useful for trial-averaged decoding where you want
to feed the model the mean of N same-label samples instead of one.

```python
from pnpl.datasets import GroupedDataset, LibriBrainPhoneme

base = LibriBrainPhoneme(data_path="./data/LibriBrain", partition="train")
grouped = GroupedDataset(base, grouped_samples=10, average_grouped_samples=True)
```

## HDF5Dataset (legacy base)

`pnpl.datasets.hdf5.HDF5Dataset` is the older base for datasets whose
data is already serialized as H5 on disk (no download). New datasets
should compose the mixins above instead.
