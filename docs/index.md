---
title: PNPL
---

# Aloha to the PNPL library! 🍍

PNPL is a friendly Python toolkit for loading and processing brain
datasets for deep learning. It ships PyTorch `Dataset` classes for
several public MEG corpora — LibriBrain, MEG-MASC (Gwilliams 2022),
Armeni 2022, and the MOUS corpus (Schöffelen 2019) — together with a
composable preprocessing pipeline and task abstractions so you can
focus on modeling, not file plumbing.

- [PyPI: `pnpl`](https://pypi.org/project/pnpl/)
- [GitHub: Neural-Processing-Lab/pnpl](https://github.com/neural-processing-lab/pnpl)


## Get Started

1) Install PNPL

```bash
pip install pnpl
```

2) Load LibriBrain with the task-based API

```python
from pnpl.datasets import LibriBrain
from pnpl.tasks import SpeechDetection

ds = LibriBrain(
    data_path="./data/LibriBrain",
    task=SpeechDetection(tmin=0.0, tmax=0.2),
    partition="train",
)
x, y, info = ds[0]
print(x.shape, y.shape)  # (channels,time), (time,)
```

The same task-based pattern works for the other datasets
(`Gwilliams2022`, `Armeni2022`, `Schoffelen2019`); they each take a
matching task object from `pnpl.tasks.<dataset>` and an optional
preprocessing string. Wrapper classes such as `LibriBrainSpeech` and
`LibriBrainPhoneme` are also available.

## Explore PNPL

<div class="feature-grid">

  <div class="feature-card">
    <div class="kicker">Start here</div>
    <h3>Quickstart</h3>
    <p>Install and load your first dataset run in a few lines.</p>
    <a class="stretched" href="quickstart.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Datasets</div>
    <h3>Dataset overview</h3>
    <p>The four shipped datasets, what each one needs, and how to pick.</p>
    <a class="stretched" href="datasets.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Pipelines</div>
    <h3>Preprocessing</h3>
    <p>The composable pipeline that turns raw MEG into cached H5.</p>
    <a class="stretched" href="preprocessing.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Pipelines</div>
    <h3>Tasks</h3>
    <p>How sample windows and labels are defined; available tasks per dataset.</p>
    <a class="stretched" href="tasks.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Reference</div>
    <h3>API</h3>
    <p>Auto‑generated docs for classes and modules with links to source.</p>
    <a class="stretched" href="api/index.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Tutorial</div>
    <h3>Speech Detection (LibriBrain)</h3>
    <p>Learn speech vs. silence classification with a compact walkthrough and Colab GPU.</p>
    <a class="stretched" href="LibriBrain_Competition_Speech_Detection.html"></a>
  </div>

  <div class="feature-card">
    <div class="kicker">Tutorial</div>
    <h3>Phoneme Classification (LibriBrain)</h3>
    <p>Build a phoneme recognizer on MEG with practical tips and code.</p>
    <a class="stretched" href="LibriBrain_Competition_Phoneme_Classification.html"></a>
  </div>

</div>

## Future Plans

While LibriBrain is still the headline use case (the [2025 LibriBrain
competition](https://libribrain.com) ships against this package),
we'll maintain `pnpl` for years to come and hope to grow it into a
useful asset for the wider community. The recent refactor adds
MEG-MASC, Armeni 2022, and the MOUS corpus alongside a shared
preprocessing pipeline; on the roadmap:

- More public datasets and dataset loaders
- Easy-to-use preprocessing pipelines
- Data augmentation options


## Contribute

We welcome issues and pull requests. See the Contributor Guide for setup and guidelines.
