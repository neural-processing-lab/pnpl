# MegNIST

MegNIST is a magnetoencephalography (MEG) dataset for benchmarking
non-invasive inner-speech decoding. It contains 12,000 trials from one
participant imagining the digits zero through nine.

The dataset is hosted on Hugging Face at `pnpl/MegNIST`. Raw data are
available in BIDS format, together with preprocessed derivatives and
machine-learning-ready HDF5 files.

## Loading the dataset

```python
from pnpl.datasets import MegNIST

dataset = MegNIST(
    data_path="./data/MegNIST",
    partition="train",
)

x, y = dataset[0]

print(x.shape)  # (306, 250)
print(y)        # digit label 0-9
```

The standard partitions are:

- `train`: 10,000 trials
- `validation` / `val`: 1,000 trials
- `test`: 1,000 trials

Each sample contains 306 MEG channels and 250 time samples at 250 Hz.

## Task

The default task is 10-class imagined-digit classification:

```python
from pnpl.datasets import MegNIST
from pnpl.tasks import DigitClassification

dataset = MegNIST(
    data_path="./data/MegNIST",
    task=DigitClassification(),
    partition="train",
)
```

Labels map directly onto the imagined digits:

| Label | Digit |
| ---: | --- |
| 0 | zero |
| 1 | one |
| 2 | two |
| 3 | three |
| 4 | four |
| 5 | five |
| 6 | six |
| 7 | seven |
| 8 | eight |
| 9 | nine |

## Standardisation

By default, MegNIST returns the released preprocessed epochs without
additional normalisation (`standardize=False`).

PNPL can optionally apply channel-wise standardisation:

```python
dataset = MegNIST(
    data_path="./data/MegNIST",
    partition="train",
    standardize=True,
)
```

Channel-wise standardisation estimates one mean and standard deviation per
MEG sensor, rather than separately for every time point. It is PNPL's
default for continuous/streaming datasets, where time points do not have a
fixed event-locked meaning and feature-wise standardisation would depend on
an arbitrary position within each window.

For fixed, event-locked epochs such as MegNIST, other choices are possible.
For example, the MegNIST baseline uses train-fitted feature-wise scaling.
If you need a different normalisation scheme, leave `standardize=False` and
apply it separately in your analysis pipeline.

## Data source

Hugging Face dataset: `pnpl/MegNIST`