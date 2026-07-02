"""
Load the LibriBrain 2026 competition holdout data and enumerate the canonical
classification examples that make up a submission.

The holdout recordings live in the public Hugging Face dataset
``pnpl/LibriBrain-Competition-2026`` under ``COMPETITION_HOLDOUT/``. Each of the
40 subjects (``subj00`` … ``subj39``) has two ``.npz`` files:

``subjXX_holdout01_sentence.npz`` -- sentence-epoched MEG
    ``meg``                    ``(N_sent, 306, T)`` float32, zero-padded in time
    ``sentence_n_times``       ``(N_sent,)``        true sample count per sentence
    ``sentence_sample_mask``   ``(N_sent, T)`` bool  valid-sample mask
    ``word_onsets_s``          ``(N_sent, W)`` float  per-word onset (s from
                                                      sentence start), NaN-padded
    ``word_mask``              ``(N_sent, W)`` bool   which word slots are valid
    ``sfreq``                  scalar (250.0 Hz)
    ``sentence_extra_end_s``   scalar -- extra tail appended after the last word
                                         onset so every word has a full window

``subjXX_holdout2_word.npz`` -- isolated word-epoched MEG (no overlap)
    ``meg``                    ``(N_word, 306, S)`` float32 -- each epoch spans
                                                              exactly ``[tmin, tmax]``
    ``sfreq``                  scalar (250.0 Hz)
    ``tmin`` / ``tmax``        scalars (0.0 / 1.0 s)

A single *classification example* is one word: for the ``word`` source it is a
stored isolated epoch; for the ``sentence`` source it is a
:data:`WINDOW_SECONDS`-second window cut from the sentence MEG starting at the
word onset. This loader turns those into a flat, deterministically ordered list
of ``(306, S)`` windows whose position defines the submission ``index``.

Canonical ordering (this loader is the source of truth -- the host builds the
solution/label CSV by iterating the *same* loader, so indices align by
construction):

1. subjects in ascending numeric order,
2. within a subject, the ``sentence`` source before the ``word`` source
   (matching the ``holdout01`` / ``holdout2`` naming),
3. within the ``sentence`` source, sentences in stored order and, inside each
   sentence, valid words (``word_mask``) in stored order,
4. within the ``word`` source, epochs in stored order.

The stored order is already shuffled per file (see ``shuffle_seed`` in each
archive), so no further shuffling is applied or wanted.

Typical usage::

    from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

    holdout = LibriBrainCompetitionHoldout(track="deep")   # subj00
    probs_primary, probs_moses = [], []
    for meg, meta in holdout.iter_windows(batch_size=256):  # meg: (B, 306, 250)
        p, m = my_model(meg)                                # your model -> (B, 50)
        probs_primary.append(p); probs_moses.append(m)

    write_submission(
        "submission_deep.csv",
        indices=holdout.indices,
        primary_probs=np.concatenate(probs_primary),
        secondary_probs=np.concatenate(probs_moses),
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


HOLDOUT_REPO = "pnpl/LibriBrain-Competition-2026"
HOLDOUT_SUBDIR = "COMPETITION_HOLDOUT"

#: Number of subjects with released holdout data (subj00 … subj39).
N_SUBJECTS = 40

#: Competition tracks -> the subjects they evaluate. ``deep`` is within-subject
#: decoding on subject 0; ``broad`` is cross-subject generalisation on the rest.
#: (The paper's Broad core is subjects 1–32 with 33–39 as zero-shot holdout
#: subjects; they share a single submission surface here. Override ``subjects``
#: if a future split differs.)
DEEP_SUBJECTS: Tuple[int, ...] = (0,)
BROAD_SUBJECTS: Tuple[int, ...] = tuple(range(1, N_SUBJECTS))
TRACKS: Dict[str, Tuple[int, ...]] = {"deep": DEEP_SUBJECTS, "broad": BROAD_SUBJECTS}

#: The two per-subject holdout sources, in canonical order.
SOURCES: Tuple[str, ...] = ("sentence", "word")
_SOURCE_SUFFIX = {"sentence": "holdout01_sentence", "word": "holdout2_word"}

#: Length (seconds) of a word window cut from the sentence MEG. Matches the
#: isolated ``holdout2_word`` epochs (``tmax - tmin == 1.0``).
WINDOW_SECONDS = 1.0


class HoldoutError(RuntimeError):
    """Raised when holdout data cannot be located, downloaded, or parsed."""


def _subject_filename(subject: int, source: str) -> str:
    return f"{HOLDOUT_SUBDIR}/subj{subject:02d}_{_SOURCE_SUFFIX[source]}.npz"


def _npz_member_shape(path: str, member: str) -> Optional[Tuple[int, ...]]:
    """Read an array's shape from an ``.npz`` without decompressing its data.

    Returns ``None`` if the header can't be parsed (caller should fall back to a
    normal load). Avoids decompressing the large ``meg`` array just to count rows.
    """
    import zipfile

    from numpy.lib import format as _npformat

    name = member if member.endswith(".npy") else member + ".npy"
    try:
        with zipfile.ZipFile(path) as z, z.open(name) as f:
            version = _npformat.read_magic(f)
            shape, _fortran, _dtype = _npformat._read_array_header(f, version)
        return shape
    except Exception:  # pragma: no cover - defensive; fall back to full load
        return None


def _resolve_subjects(
    track: Optional[str], subjects: Optional[Sequence[int]]
) -> Tuple[int, ...]:
    if track is not None and subjects is not None:
        raise HoldoutError("Pass either `track` or `subjects`, not both.")
    if track is not None:
        key = track.lower()
        if key not in TRACKS:
            raise HoldoutError(
                f"Unknown track {track!r}; expected one of {sorted(TRACKS)}."
            )
        return TRACKS[key]
    if subjects is not None:
        subs = tuple(int(s) for s in subjects)
        if not subs:
            raise HoldoutError("`subjects` is empty.")
        for s in subs:
            if not 0 <= s < N_SUBJECTS:
                raise HoldoutError(
                    f"Subject {s} out of range 0..{N_SUBJECTS - 1}."
                )
        return subs
    raise HoldoutError("Pass a `track` (e.g. 'deep'/'broad') or an explicit `subjects` list.")


class LibriBrainCompetitionHoldout:
    """Enumerate and serve the LibriBrain 2026 competition holdout windows.

    Parameters
    ----------
    track:
        ``"deep"`` (subject 0) or ``"broad"`` (subjects 1–39). Mutually
        exclusive with ``subjects``.
    subjects:
        Explicit list of subject ids to include (overrides ``track``).
    sources:
        Which per-subject holdout sources to enumerate. Defaults to both
        ``("sentence", "word")`` in canonical order. Restrict to a single
        source (e.g. ``("word",)``) if you only need one.
    data_path:
        Directory to download the ``.npz`` files into. When ``None`` (default)
        the shared Hugging Face cache is used.
    download:
        If ``True`` (default), missing files are fetched from
        :data:`HOLDOUT_REPO`. If ``False``, files must already be present.
    hf_token:
        Optional Hugging Face token (the dataset is public, so normally
        unnecessary). Falls back to ``HF_TOKEN`` in the environment.

    Attributes
    ----------
    subjects:
        Tuple of included subject ids.
    metadata:
        List of per-example dicts (length ``N``), each with keys ``index``,
        ``subject``, ``source``, ``epoch``, ``word``, ``onset_s``,
        ``start_sample``. ``word``/``onset_s`` are meaningful for the sentence
        source; for the word source ``word`` is ``-1`` and ``onset_s`` is
        ``0.0``.
    indices:
        ``np.arange(N)`` -- the canonical submission indices.
    """

    def __init__(
        self,
        track: Optional[str] = None,
        subjects: Optional[Sequence[int]] = None,
        *,
        sources: Sequence[str] = SOURCES,
        data_path: Optional[Union[str, Path]] = None,
        download: bool = True,
        hf_token: Optional[str] = None,
    ) -> None:
        self.subjects = _resolve_subjects(track, subjects)
        self.sources = tuple(sources)
        for s in self.sources:
            if s not in _SOURCE_SUFFIX:
                raise HoldoutError(f"Unknown source {s!r}; expected 'sentence'/'word'.")
        self.data_path = str(Path(data_path).expanduser()) if data_path is not None else None
        self.download = download
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Resolve (download if needed) every file we'll touch, then build the
        # flat example index from the lightweight metadata arrays only.
        self._file_paths: Dict[Tuple[int, str], str] = {}
        self.metadata: List[dict] = []
        self._build_index()
        self.indices = np.arange(len(self.metadata), dtype=np.int64)

        # One-file MEG cache (keyed by resolved path). Sequential access -- e.g.
        # `iter_windows` or a non-shuffled DataLoader -- keeps this at ~1 miss
        # per file. Random access still works, just re-loads on file changes.
        self._cache_path: Optional[str] = None
        self._cache_meg: Optional[np.ndarray] = None

    # -- construction -----------------------------------------------------

    def _ensure_file(self, subject: int, source: str) -> str:
        key = (subject, source)
        cached = self._file_paths.get(key)
        if cached is not None:
            return cached
        rel = _subject_filename(subject, source)
        if self.data_path is not None:
            local = os.path.join(self.data_path, rel)
            if os.path.exists(local):
                self._file_paths[key] = local
                return local
        if not self.download:
            where = os.path.join(self.data_path or "<hf-cache>", rel)
            raise HoldoutError(
                f"Missing holdout file for subj{subject:02d} ({source}): {where}. "
                "Enable download=True or provide the file locally."
            )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise HoldoutError(
                "huggingface_hub is required to download holdout data. "
                "Install it with `pip install huggingface_hub`."
            ) from e
        try:
            path = hf_hub_download(
                repo_id=HOLDOUT_REPO,
                filename=rel,
                repo_type="dataset",
                token=self.hf_token,
                local_dir=self.data_path,
            )
        except Exception as e:  # pragma: no cover - network/permission errors
            raise HoldoutError(
                f"Failed to download {rel} from {HOLDOUT_REPO}: {e}"
            ) from e
        self._file_paths[key] = path
        return path

    def _build_index(self) -> None:
        win = None
        for subject in self.subjects:
            for source in self.sources:
                path = self._ensure_file(subject, source)
                with np.load(path, allow_pickle=True) as d:
                    sfreq = float(d["sfreq"])
                    w = int(round(sfreq * WINDOW_SECONDS))
                    if win is None:
                        win = w
                    elif w != win:
                        raise HoldoutError(
                            f"Inconsistent window length: {w} vs {win} samples "
                            f"(subj{subject:02d} {source}, sfreq={sfreq})."
                        )
                    if source == "sentence":
                        self._index_sentence_file(subject, d, sfreq, w)
                    else:
                        self._index_word_file(subject, path, d, w)
        self._window_samples = win if win is not None else int(round(250.0 * WINDOW_SECONDS))

    def _index_sentence_file(self, subject: int, d, sfreq: float, win: int) -> None:
        word_mask = np.asarray(d["word_mask"])
        word_onsets = np.asarray(d["word_onsets_s"], dtype=np.float64)
        n_times = np.asarray(d["sentence_n_times"]).astype(np.int64)
        n_sent = word_mask.shape[0]
        for si in range(n_sent):
            valid = np.nonzero(word_mask[si])[0]
            for wi in valid:
                onset = float(word_onsets[si, wi])
                start = int(round(onset * sfreq))
                if start < 0 or start + win > int(n_times[si]):
                    # Should never happen (sentence_extra_end_s guarantees a
                    # full window), but guard rather than emit padded samples.
                    raise HoldoutError(
                        f"subj{subject:02d} sentence {si} word {wi}: window "
                        f"[{start}:{start + win}] exceeds n_times={int(n_times[si])}."
                    )
                self.metadata.append(
                    {
                        "index": len(self.metadata),
                        "subject": subject,
                        "source": "sentence",
                        "epoch": si,
                        "word": int(wi),
                        "onset_s": onset,
                        "start_sample": start,
                    }
                )

    def _index_word_file(self, subject: int, path: str, d, win: int) -> None:
        # Count epochs from the .npy header so we don't decompress the full meg.
        shape = _npz_member_shape(path, "meg")
        n_epochs = int(shape[0]) if shape else int(np.asarray(d["meg"]).shape[0])
        for ei in range(n_epochs):
            self.metadata.append(
                {
                    "index": len(self.metadata),
                    "subject": subject,
                    "source": "word",
                    "epoch": ei,
                    "word": -1,
                    "onset_s": 0.0,
                    "start_sample": 0,
                }
            )

    # -- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_meg(self, subject: int, source: str) -> np.ndarray:
        path = self._ensure_file(subject, source)
        if path != self._cache_path:
            with np.load(path, allow_pickle=True) as d:
                self._cache_meg = np.asarray(d["meg"], dtype=np.float32)
            self._cache_path = path
        return self._cache_meg  # type: ignore[return-value]

    def _window_from_meta(self, meta: dict) -> np.ndarray:
        meg = self._load_meg(meta["subject"], meta["source"])
        epoch = meg[meta["epoch"]]
        start = meta["start_sample"]
        return np.ascontiguousarray(epoch[:, start : start + self._window_samples])

    def __getitem__(self, i: int) -> Tuple[np.ndarray, dict]:
        """Return ``(window, meta)`` for row ``i``.

        ``window`` is a ``(306, window_samples)`` float32 array; ``meta`` is the
        dict from :attr:`metadata`.
        """
        if i < 0:
            i += len(self.metadata)
        if not 0 <= i < len(self.metadata):
            raise IndexError(i)
        meta = self.metadata[i]
        return self._window_from_meta(meta), meta

    def iter_windows(
        self, batch_size: Optional[int] = None
    ) -> Iterator[Tuple[np.ndarray, Union[dict, List[dict]]]]:
        """Iterate windows in canonical order, one MEG file resident at a time.

        With ``batch_size=None`` yields ``(window (306, S), meta dict)`` per
        example. With an integer ``batch_size`` yields
        ``(batch (B, 306, S), [meta, ...])``. Batching is purely for throughput:
        a batch may span subjects/sources (always in canonical order), and the
        final batch may be smaller than ``batch_size``.
        """
        if batch_size is not None and batch_size <= 0:
            raise HoldoutError("batch_size must be a positive integer or None.")
        buf_w: List[np.ndarray] = []
        buf_m: List[dict] = []
        for meta in self.metadata:
            window = self._window_from_meta(meta)
            if batch_size is None:
                yield window, meta
                continue
            buf_w.append(window)
            buf_m.append(meta)
            if len(buf_w) == batch_size:
                yield np.stack(buf_w), list(buf_m)
                buf_w, buf_m = [], []
        if batch_size is not None and buf_w:
            yield np.stack(buf_w), list(buf_m)

    def metadata_frame(self):
        """Return the example metadata as a :class:`pandas.DataFrame`."""
        import pandas as pd  # local import: keep pandas optional

        return pd.DataFrame(self.metadata)

    def counts(self) -> Dict[str, int]:
        """Return example counts, e.g. ``{'total': N, 'sentence': .., 'word': ..}``."""
        out = {"total": len(self.metadata)}
        for src in self.sources:
            out[src] = sum(1 for m in self.metadata if m["source"] == src)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        subs = (
            f"{self.subjects[0]}..{self.subjects[-1]}"
            if len(self.subjects) > 3
            else ",".join(map(str, self.subjects))
        )
        return (
            f"LibriBrainCompetitionHoldout(subjects=[{subs}], "
            f"sources={self.sources}, n_examples={len(self.metadata)})"
        )
