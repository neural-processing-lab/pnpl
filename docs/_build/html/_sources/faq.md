---
title: FAQ
---

# Frequently Asked Questions

## Why does the first dataset load take longer?
If `download=True`, required files are fetched from Hugging Face and cached under `data_path`. Subsequent runs reuse the local cache.

## How do I load just a small subset quickly?
Pick a single run key from `pnpl.datasets.libribrain2025.constants.RUN_KEYS` and pass it via `include_run_keys=[...]`.

## How do I standardize data?
Datasets standardize by default using per‑run channel stats. Disable with `standardize=False` or provide `channel_means`/`channel_stds` explicitly.

## How do I cite PNPL?
See the Citation page (coming soon). For now, please link to the project repository.

