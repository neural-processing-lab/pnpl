"""Offline tests for pnpl.competition.holdout (synthetic npz fixtures)."""

import numpy as np
import pytest

from pnpl.competition import (
    LibriBrainCompetitionHoldout,
    HoldoutError,
    write_submission,
    PRIMARY_VOCAB,
    SECONDARY_VOCAB,
)
from pnpl.competition.holdout import HOLDOUT_SUBDIR, WINDOW_SECONDS


SFREQ = 250.0
WIN = int(round(SFREQ * WINDOW_SECONDS))  # 250
N_CH = 306


def _write_sentence_file(path, *, n_sent, words_per_sent, seed, valid_full_window=True):
    """Write a synthetic holdout01_sentence npz with a deterministic MEG signal.

    meg[si, ch, t] = si*1000 + ch + t/1000  so every extracted window is unique
    and cheap to cross-check.
    """
    rng = np.random.default_rng(seed)
    W = max(words_per_sent)
    T = 1030
    meg = np.zeros((n_sent, N_CH, T), dtype=np.float32)
    for si in range(n_sent):
        t = np.arange(T)
        meg[si] = (si * 1000 + np.arange(N_CH)[:, None] + t[None, :] / 1000.0).astype(np.float32)
    word_onsets = np.full((n_sent, W), np.nan, dtype=np.float32)
    word_mask = np.zeros((n_sent, W), dtype=bool)
    n_times = np.zeros(n_sent, dtype=np.int64)
    for si in range(n_sent):
        k = words_per_sent[si]
        onsets = np.sort(rng.uniform(0.0, 1.5, size=k)).astype(np.float32)
        word_onsets[si, :k] = onsets
        word_mask[si, :k] = True
        last_start = int(round(float(onsets[-1]) * SFREQ))
        # extra_end guarantees a full window past the last onset
        n_times[si] = last_start + WIN if valid_full_window else last_start + WIN - 5
    # zero out padded region beyond n_times (mirrors the real files)
    for si in range(n_sent):
        meg[si, :, int(n_times[si]):] = 0.0
    sample_mask = np.zeros((n_sent, T), dtype=bool)
    for si in range(n_sent):
        sample_mask[si, : int(n_times[si])] = True
    np.savez(
        path,
        meg=meg,
        sentence_n_times=n_times,
        sentence_sample_mask=sample_mask,
        word_onsets_s=word_onsets,
        word_mask=word_mask,
        sfreq=np.float64(SFREQ),
        sentence_extra_end_s=np.float64(1.0),
        shuffle_seed=np.int64(seed),
    )


def _write_word_file(path, *, n_word, seed):
    meg = (np.arange(n_word)[:, None, None] * 7.0
           + np.arange(N_CH)[None, :, None]
           + np.arange(WIN)[None, None, :] / 1000.0).astype(np.float32)
    np.savez(
        path,
        meg=meg,
        sfreq=np.float64(SFREQ),
        tmin=np.float64(0.0),
        tmax=np.float64(1.0),
        shuffle_seed=np.int64(seed),
    )


@pytest.fixture
def holdout_root(tmp_path):
    """A data_path with COMPETITION_HOLDOUT/subj00 + subj01 synthetic files."""
    d = tmp_path / HOLDOUT_SUBDIR
    d.mkdir()
    # subj00: 3 sentences with 2,1,3 words -> 6 sentence words; 4 isolated words
    _write_sentence_file(d / "subj00_holdout01_sentence.npz",
                         n_sent=3, words_per_sent=[2, 1, 3], seed=100)
    _write_word_file(d / "subj00_holdout2_word.npz", n_word=4, seed=200)
    # subj01: 2 sentences with 3,2 words -> 5 sentence words; 2 isolated words
    _write_sentence_file(d / "subj01_holdout01_sentence.npz",
                         n_sent=2, words_per_sent=[3, 2], seed=101)
    _write_word_file(d / "subj01_holdout2_word.npz", n_word=2, seed=201)
    return tmp_path


def test_enumeration_counts_and_order(holdout_root):
    h = LibriBrainCompetitionHoldout(subjects=[0], data_path=str(holdout_root), download=False)
    assert len(h) == 6 + 4
    assert h.counts() == {"total": 10, "sentence": 6, "word": 4}
    # sentence rows come first, in (sentence, word) order, then word rows
    srcs = [m["source"] for m in h.metadata]
    assert srcs == ["sentence"] * 6 + ["word"] * 4
    # indices are 0..N-1 and match metadata["index"]
    assert list(h.indices) == list(range(10))
    assert [m["index"] for m in h.metadata] == list(range(10))
    # word rows carry word=-1, onset 0
    for m in h.metadata:
        if m["source"] == "word":
            assert m["word"] == -1 and m["onset_s"] == 0.0


def test_multi_subject_order(holdout_root):
    h = LibriBrainCompetitionHoldout(subjects=[0, 1], data_path=str(holdout_root), download=False)
    # subj00 (10) fully before subj01 (7)
    subs = [m["subject"] for m in h.metadata]
    assert subs == [0] * 10 + [1] * 7
    assert len(h) == 17


def test_windows_match_manual_extraction(holdout_root):
    h = LibriBrainCompetitionHoldout(subjects=[0], data_path=str(holdout_root), download=False)
    sent = np.load(holdout_root / HOLDOUT_SUBDIR / "subj00_holdout01_sentence.npz")
    word = np.load(holdout_root / HOLDOUT_SUBDIR / "subj00_holdout2_word.npz")
    for i in range(len(h)):
        win, meta = h[i]
        assert win.shape == (N_CH, WIN) and win.dtype == np.float32
        if meta["source"] == "sentence":
            si, wi = meta["epoch"], meta["word"]
            start = int(round(float(sent["word_onsets_s"][si, wi]) * SFREQ))
            assert start == meta["start_sample"]
            assert np.array_equal(win, sent["meg"][si, :, start:start + WIN])
            assert win.any()  # no all-zero (padding) window
        else:
            assert np.array_equal(win, word["meg"][meta["epoch"]])


def test_iter_windows_batching(holdout_root):
    h = LibriBrainCompetitionHoldout(subjects=[0, 1], data_path=str(holdout_root), download=False)
    # unbatched
    rows = list(h.iter_windows())
    assert len(rows) == len(h)
    assert all(w.shape == (N_CH, WIN) for w, _ in rows)
    # batched
    total = 0
    for batch, metas in h.iter_windows(batch_size=4):
        assert batch.ndim == 3 and batch.shape[1:] == (N_CH, WIN)
        assert batch.shape[0] == len(metas)
        total += batch.shape[0]
    assert total == len(h)


def test_track_resolution_errors(holdout_root):
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(track="deep", subjects=[0], data_path=str(holdout_root))
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(track="nope", data_path=str(holdout_root))
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(data_path=str(holdout_root))
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(subjects=[999], data_path=str(holdout_root))


def test_missing_file_without_download(tmp_path):
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(subjects=[0], data_path=str(tmp_path), download=False)


def test_window_exceeding_n_times_raises(tmp_path):
    d = tmp_path / HOLDOUT_SUBDIR
    d.mkdir()
    _write_sentence_file(d / "subj00_holdout01_sentence.npz",
                         n_sent=2, words_per_sent=[2, 2], seed=7, valid_full_window=False)
    _write_word_file(d / "subj00_holdout2_word.npz", n_word=1, seed=8)
    with pytest.raises(HoldoutError):
        LibriBrainCompetitionHoldout(subjects=[0], data_path=str(tmp_path), download=False)


def test_sources_restriction(holdout_root):
    h = LibriBrainCompetitionHoldout(subjects=[0], sources=("word",),
                                     data_path=str(holdout_root), download=False)
    assert h.counts() == {"total": 4, "word": 4}
    assert all(m["source"] == "word" for m in h.metadata)


def test_integration_with_write_submission(holdout_root, tmp_path):
    h = LibriBrainCompetitionHoldout(subjects=[0, 1], data_path=str(holdout_root), download=False)
    n = len(h)
    primary = np.full((n, len(PRIMARY_VOCAB)), 1.0 / len(PRIMARY_VOCAB))
    secondary = np.full((n, len(SECONDARY_VOCAB)), 1.0 / len(SECONDARY_VOCAB))
    out = write_submission(tmp_path / "sub.csv", indices=h.indices,
                           primary_probs=primary, secondary_probs=secondary)
    import csv
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "index"
    assert len(rows[0]) == 1 + len(PRIMARY_VOCAB) + len(SECONDARY_VOCAB)
    assert len(rows) == n + 1  # header + N
    assert [int(r[0]) for r in rows[1:]] == list(range(n))
