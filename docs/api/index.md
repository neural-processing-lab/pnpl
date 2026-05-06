---
title: API Reference
---

# API Reference

Below you'll find the public Python API, auto‑generated from
docstrings. The summaries link to detailed pages for each class or
module.

## Datasets

```{eval-rst}
.. autosummary::
   :toctree: generated
   :caption: Datasets
   :nosignatures:

   pnpl.datasets.libribrain2025.dataset.LibriBrain
   pnpl.datasets.libribrain2025.compat.LibriBrainSpeech
   pnpl.datasets.libribrain2025.compat.LibriBrainPhoneme
   pnpl.datasets.libribrain2025.compat.LibriBrainWord
   pnpl.datasets.libribrain2025.sentence_dataset.LibriBrainSentence
   pnpl.datasets.libribrain100.dataset.LibriBrain100
   pnpl.datasets.libribrain100.compat.LibriBrain100Speech
   pnpl.datasets.libribrain100.compat.LibriBrain100Phoneme
   pnpl.datasets.libribrain100.compat.LibriBrain100Word
   pnpl.datasets.gwilliams2022.dataset.Gwilliams2022
   pnpl.datasets.armeni2022.dataset.Armeni2022
   pnpl.datasets.schoffelen2019.dataset.Schoffelen2019
   pnpl.datasets.pallier2025.dataset.Pallier2025
   pnpl.datasets.grouped_dataset.GroupedDataset
   pnpl.datasets.hdf5.dataset.HDF5Dataset
```

## Tasks

```{eval-rst}
.. autosummary::
   :toctree: generated
   :caption: Tasks
   :nosignatures:

   pnpl.tasks.base.TaskProtocol
   pnpl.tasks.libribrain.SpeechDetection
   pnpl.tasks.libribrain.PhonemeClassification
   pnpl.tasks.libribrain.WordClassification
   pnpl.tasks.gwilliams2022.PhonemeClassification
   pnpl.tasks.gwilliams2022.WordClassification
   pnpl.tasks.armeni2022.PhonemeClassification
   pnpl.tasks.schoffelen2019.TrialEpoching
   pnpl.tasks.pallier2025.WordClassification
```

## Preprocessing

```{eval-rst}
.. autosummary::
   :toctree: generated
   :caption: Preprocessing
   :nosignatures:

   pnpl.preprocessing.Pipeline
   pnpl.preprocessing.BadChannels
   pnpl.preprocessing.HeadPosition
   pnpl.preprocessing.MaxwellFilter
   pnpl.preprocessing.NotchFilter
   pnpl.preprocessing.BandpassFilter
   pnpl.preprocessing.Downsample
   pnpl.preprocessing.Epoch
   pnpl.preprocessing.fif_to_h5
   pnpl.preprocessing.epochs_to_h5
```

## Mixins

```{eval-rst}
.. autosummary::
   :toctree: generated
   :caption: Mixins (for new datasets)
   :nosignatures:

   pnpl.datasets.mixins.HFDownloadMixin
   pnpl.datasets.mixins.OSFDownloadMixin
   pnpl.datasets.mixins.RadboudDownloadMixin
   pnpl.datasets.mixins.OpenNeuroDownloadMixin
   pnpl.datasets.mixins.BIDSMixin
   pnpl.datasets.mixins.ContinuousH5Mixin
   pnpl.datasets.mixins.EpochedH5Mixin
   pnpl.datasets.mixins.StandardizationMixin
```

## LibriBrain Constants

```{eval-rst}
.. autosummary::
   :toctree: generated
   :caption: LibriBrain Utilities / Constants
   :nosignatures:

   pnpl.datasets.libribrain2025.constants
```
