"""
Load LibriBrain100 — minimal end-to-end example.

LibriBrain100 is the unified loader for the full LibriBrain release: a
virtual union of pnpl/LibriBrain (sub-0 × Sherlock1..7) and
pnpl/LibriBrain2 (sub-0 × Sherlock8/Sherlock9/TIMIT/MOCHATIMIT/
TheMoth, plus sub-1..32 × Sherlock1 ses-11/ses-12). The library
fetches each record from whichever underlying repo owns it.

Run:
    python examples/load_libribrain100.py
"""

from __future__ import annotations

from pnpl.datasets import LibriBrain100, LibriBrain100Word
from pnpl.tasks import SpeechDetection


def task_based_example():
    """Task-based entry point — pass any TaskProtocol object."""
    print("=== TASK-BASED ENTRY POINT ===\n")

    ds = LibriBrain100(
        data_path="./data/LibriBrain100",
        task=SpeechDetection(tmin=0.0, tmax=0.5),
        partition="validation",       # Sherlock1 ses-11
        subjects="deep",              # subject 0 only
        corpus="sherlock",
        download=True,
    )
    print(f"Samples:        {len(ds)}")
    print(f"n_channels:     {ds.n_channels}")
    print(f"n_times/sample: {ds.n_times}")

    x, y = ds[0]
    print(f"First sample: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    print()


def selectors_example():
    """Subset the download to one corpus on the deep subject."""
    print("=== SELECTORS ===\n")

    ds = LibriBrain100Word(
        data_path="./data/LibriBrain100",
        subjects="deep",
        corpus="podcasts",            # 30 'The Moth' stories
        include_run_keys=[("0", "1", "TheMoth", "1")],
        download=True,
    )
    print(f"Samples:    {len(ds)}")
    print(f"Vocab size: {ds.label_info['n_classes']}")

    x, y = ds[0]
    id_to_word = ds.label_info["id_to_label"]
    print(f"First sample: x.shape={tuple(x.shape)}, y={int(y)} ('{id_to_word[int(y)]}')")
    print()


if __name__ == "__main__":
    task_based_example()
    selectors_example()
