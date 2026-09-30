# Clinical communication benchmark

This loader reproduces the frozen, single-subject LibriBrain clinical
communication benchmark. It bundles sentence text, recording partitions, event
row IDs, and the ordered five-occurrence donor assignments. It never generates
new sentences or resamples occurrences.

## Usage

```python
from pnpl.datasets import ClinicalCommunication

root = "./data/LibriBrain100"
train = ClinicalCommunication(root, partition="train")
val = ClinicalCommunication(root, partition="val")
dev = ClinicalCommunication(root, partition="dev", k=5)  # optional 50 sentences
test = ClinicalCommunication(root, partition="test", k=5)  # original 100 sentences

sample = test[0]
# sample["meg"]: (4 words, 5 occurrences, 306 channels, 150 samples)
# sample["words"]: ["can", "you", "help", "me"]
# sample["sentence"]: "Can you help me?"
# sample["occurrences"][word][member]: recording, event_index, onset, etc.
```

Use `test_sentences=200` to opt into the extended test set. The first 100 sentences
and every one of their donors remain unchanged. `k` is an integer from 1 to 5:
each word uses the first k members of its frozen assignment, in the original
order. These are the nested subsets used in the original k curve, not all
combinations of donors. Windows are returned separately; the loader does not
average signals or model predictions.

No embeddings, language model, decoder, or model weights are required.
`candidate_words` exposes the fixed 92-word clinical vocabulary.

## Train, validation and development

| Partition | Recordings | Natural word windows | Clinical sentences |
|---|---:|---:|---:|
| train | 121 | 595,598 | — |
| val / validation | 10 | 51,126 | — |
| dev / development | Uses val recordings | 307 positions × k | 50 |
| test | 31 | 595 positions × k | 100 (default) |
| test, test_sentences=200 | Same 31 recordings | 1,275 positions × k | 200 |

All recordings are subject 0. Train sessions are 1–6, 8–10, 12, 17, 19, and
23–28. Validation is session 11. Test sessions are 0, 7, 13–16, 18, 20–22, 29,
and 30. These benchmark splits override LibriBrain100's general-purpose split.

Train/val default to individual natural word windows. Set `unit="sentence"`
for the original natural sentence groups, or `unit="group"` for the frozen
disjoint same-word groups used by the five-occurrence baseline (111,917 train
groups and 6,845 val groups). The latter selects the first k members too.
The natural-word view includes the full prepared occurrence inventory, whereas
the grouped view preserves the original baseline's vocabulary filtering and
unused remainders.

The 50 development sentences use the **original validation-group assignments**.
They are a view of validation data, not an independent recording partition.
They overlap validation windows intentionally. Test donor recordings are
disjoint from train/val. All donors within each clinical sentence set are unique.

## Returned items

Items are dictionaries containing:

- `meg`: float32 tensor with shapes below.
- `words`: ordered lower-case target word strings.
- `sentence`: original clinical sentence text, or joined natural words.
- `sentence_id`: original one-based clinical ID; natural sentence index for
  the natural sentence view; otherwise None.
- `occurrences`: nested list, indexed by word and occurrence member. Each
  donor includes its portable source path, published event row index, subject,
  session, recording-relative onset, and published onset.
- `positions`: canonical 306 × 2 sensor positions.
- `partition` and `source_split`: dev has source_split="val".

| View | meg shape |
|---|---|
| Natural word | (306, 150) |
| Natural sentence | (words, 306, 150) |
| Frozen word group | (k, 306, 150) |
| Clinical sentence (test/dev) | (words, k, 306, 150) |

Sentence lengths vary. Use `DataLoader(dataset, batch_size=None)` or a custom
padding/collation function. Labels and occurrence metadata also vary in length.
`metadata(i)` returns the metadata without loading any MEG.

## Downloads, preprocessing and cache

Construction reads only bundled metadata. Accessing a sample lazily downloads
its required recordings from the pinned pnpl/LibriBrain and pnpl/LibriBrain2
revisions. Existing files use the same corpus-relative layout as LibriBrain100.
Set `download=False` for offline use; missing sources then raise an error.
`prepare()` precomputes the selected view's recordings.

The source HDF5 and event TSV size/checksums are verified. Additional processing
matches the benchmark: filter 0.1–40 Hz, resample to 50 Hz, apply channelwise
recording-level RobustScaler, extract a 3-second window using the rounded
recording-relative onset, convert to float32, subtract the first 25 samples'
mean, and clamp to [-5, 5]. Published recording time origins are subtracted
before selecting windows. Channel order is checked against the canonical
sensor positions.

Scaled continuous float64 recordings are cached under
`data_path/.clinical_cache` (override with `cache_path`). This avoids storing
hundreds of thousands of overlapping windows. First access can be slow and
requires enough RAM for one continuous recording. At most two recording
memory maps remain open per loader process. Use `close()` to release them.
Cache identities include source hashes, the preprocessing definition, and
library versions; completed cache files are checksummed. Changes in numerical
library versions produce a separate cache. Exact floating-point values may
vary between library versions even though the occurrence identities are fixed.

The bundled provenance manifest records hashes of the original experiment
plan, occurrence CSV, and development definition. Metadata tests verify
the original counts, disjoint recording splits, donor uniqueness and order,
and identical first-100 selections in the 200-sentence extension.
