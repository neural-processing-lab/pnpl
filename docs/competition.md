# Competition submissions (PNPL 2026)

`pnpl.competition` turns your model's predictions into a valid PNPL 2026 Kaggle
submission and uploads it. There are three pieces:

1. **`LibriBrainCompetitionHoldout`** — downloads the holdout MEG and enumerates
   the canonical rows of a submission (one row per word).
2. **`write_submission`** — writes those rows + your probabilities to CSV in the
   exact leaderboard format.
3. **`submit_to_kaggle`** — uploads the CSV via the Kaggle CLI.

## The submission format

```
index, <50 primary-vocab probs>, moses_<word> × 50
```

- The first 50 probability columns are a distribution over the **competition
  vocabulary** (`load_vocabulary("primary")`), in vocab order.
- The next 50 columns are a distribution over the **Moses-50** vocabulary
  (`load_vocabulary("moses")`), each prefixed `moses_`.
- Scoring is **Top-10 Balanced Accuracy** over the primary distribution; the
  Moses block is tracked as a secondary metric. Provide full probability
  distributions (not argmax) so Top-10 can be computed.

## The holdout data

The holdout lives in the public dataset
[`pnpl/LibriBrain-Competition-2026`](https://huggingface.co/datasets/pnpl/LibriBrain-Competition-2026)
(`COMPETITION_HOLDOUT/`). Each of the 40 subjects (`subj00`–`subj39`) has two
files:

| file | contents |
| --- | --- |
| `subjXX_holdout01_sentence.npz` | sentence-epoched MEG `(N, 306, T)` (zero-padded in time) plus `word_onsets_s` / `word_mask` marking each word |
| `subjXX_holdout2_word.npz` | isolated 1 s word epochs `(N, 306, 250)` |

`LibriBrainCompetitionHoldout` expands both into `(306, 250)` windows (1 s @
250 Hz): for the sentence source it cuts a 1 s window at each word onset; for the
word source it uses the stored epoch directly.

### Tracks

| track | subjects | meaning |
| --- | --- | --- |
| `"deep"` | subject 0 | within-subject decoding |
| `"broad"` | subjects 1–39 | cross-subject generalisation |

Each track is submitted as its own CSV.

### Canonical row order

The loader **is** the source of truth for `index`. Rows are ordered:
subjects ascending → per subject the `sentence` source then the `word` source →
sentences/epochs in stored order → words within a sentence in stored order. Every
valid word (`word_mask`) becomes a row. Always pass `holdout.indices` to
`write_submission` and never reorder your predictions.

## End-to-end

```python
import numpy as np
from pnpl.competition import (
    LibriBrainCompetitionHoldout, write_submission, submit_to_kaggle,
)

holdout = LibriBrainCompetitionHoldout(track="deep")   # subj00; downloads on first use
print(len(holdout), holdout.counts())                  # 960 {'total':960,'sentence':868,'word':92}

primary, secondary = [], []
for meg, metas in holdout.iter_windows(batch_size=256):  # meg: (B, 306, 250)
    p, m = my_model(meg)                                 # -> (B, 50), (B, 50)
    primary.append(p); secondary.append(m)

csv_path = write_submission(
    "submission_deep.csv",
    indices=holdout.indices,
    primary_probs=np.concatenate(primary),
    secondary_probs=np.concatenate(secondary),
)

submit_to_kaggle(csv_path, competition="<slug>", message="baseline")
```

`iter_windows()` keeps only one subject's MEG in memory at a time, so the large
`broad` track streams comfortably. For random access / a PyTorch `DataLoader`,
`holdout[i]` returns `(window, meta)`; use `shuffle=False` at inference so the
one-file cache stays warm.

### Kaggle auth

`submit_to_kaggle` shells out to the official Kaggle CLI, so any standard auth
works: `KAGGLE_API_TOKEN` (modern `KGAT_…` token, CLI ≥ 2.0),
`KAGGLE_USERNAME`/`KAGGLE_KEY`, or `~/.kaggle/kaggle.json`.

See the runnable [`examples/submit_pnpl2026.ipynb`](https://github.com/neural-processing-lab/pnpl-public/blob/main/examples/submit_pnpl2026.ipynb).
