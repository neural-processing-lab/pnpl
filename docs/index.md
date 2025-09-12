---
title: PNPL
---

# Aloha to the PNPL library! 🍍

PNPL is a friendly Python toolkit for loading and processing brain datasets for deep learning. It ships with ready‑to‑use dataset classes (PyTorch `Dataset`) and simple utilities so you can focus on modeling, not file plumbing.

- [PyPI: `pnpl`](https://pypi.org/project/pnpl/)
- [GitHub: Neural-Processing-Lab/pnpl](https://github.com/neural-processing-lab/pnpl-public)


## Get Started

1) Install PNPL

```bash
pip install pnpl
```

2) Load a single run of LibriBrain Speech and iterate samples

```python
from pnpl.datasets.libribrain2025 import constants
from pnpl.datasets import LibriBrainSpeech

ds = LibriBrainSpeech(
    data_path="./data/LibriBrain",
    preprocessing_str="bads+headpos+sss+notch+bp+ds",
    include_run_keys=[constants.RUN_KEYS[0]]
    )
x, y, info = ds[0]
print(x.shape, y.shape)  # (channels,time), (time,)
```

## Explore PNPL

<div class="feature-grid">


  <div class="feature-card">
    <div class="kicker">Start here</div>
    <h3>Quickstart</h3>
    <p>Install and load your first dataset run in a few lines.</p>
    <a class="stretched" href="quickstart.html"></a>
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
While currently the primary use of the PNPL library is for the 2025 LibriBrain competition (click [here](https://libribrain.com) to learn more!), we will maintain it for years to come and hope to turn it into a useful asset for the community. Among other things, that will mean adding:

- More public datasets and dataset loaders
- Easy-to-use preprocessing pipelines
- Data augmentation options


## Contribute

We welcome issues and pull requests. See the Contributor Guide for setup and guidelines.
