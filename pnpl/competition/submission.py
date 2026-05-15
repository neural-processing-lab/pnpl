"""
Build and upload Kaggle submissions for the PNPL 2026 competition.

The submission format is::

    index, <primary vocab probs...>, moses_<word> probs...

* The first 50 probability columns hold a distribution over the primary
  vocabulary (``vocabulary.csv``), in the same word order.
* The next 50 columns hold a distribution over the secondary "moses"
  vocabulary (``moses-vocabulary.csv``), prefixed with ``moses_`` so they
  remain unambiguous even when the two vocabularies share words.
* Only the primary distribution is scored (balanced accuracy on argmax);
  the secondary columns ride along for downstream use.

Typical usage::

    from pnpl.competition import write_submission, submit_to_kaggle

    write_submission(
        path="submission.csv",
        indices=indices,            # (N,)
        primary_probs=primary,      # (N, 50)
        secondary_probs=secondary,  # (N, 50)
    )

    submit_to_kaggle(
        "submission.csv",
        competition="pnpl-internal-testing",
        message="baseline",
    )

Authentication for ``submit_to_kaggle`` uses the official ``kaggle`` CLI;
set up auth via any of the standard methods (``KAGGLE_API_TOKEN`` env var
for the modern token format, ``KAGGLE_USERNAME``/``KAGGLE_KEY``, or
``~/.kaggle/kaggle.json``). See ``kaggle auth login`` for the OAuth flow.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np


SECONDARY_VOCAB_PREFIX = "moses_"
_DATA_DIR = Path(__file__).resolve().parent / "data"
_PRIMARY_VOCAB_FILE = _DATA_DIR / "vocabulary.csv"
_SECONDARY_VOCAB_FILE = _DATA_DIR / "moses-vocabulary.csv"


class SubmissionError(RuntimeError):
    """Raised when a submission cannot be built or uploaded."""


def load_vocabulary(name: str = "primary") -> list[str]:
    """Load a bundled vocabulary file as an ordered list of words.

    Parameters
    ----------
    name:
        ``"primary"`` for ``vocabulary.csv`` or ``"secondary"`` / ``"moses"``
        for ``moses-vocabulary.csv``.
    """
    if name == "primary":
        path = _PRIMARY_VOCAB_FILE
    elif name in ("secondary", "moses"):
        path = _SECONDARY_VOCAB_FILE
    else:
        raise ValueError(f"Unknown vocabulary {name!r}; expected 'primary' or 'moses'")
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    return [w.strip() for w in line.split(",") if w.strip()]


# Pre-load and cache the canonical vocabularies so calling code can compare
# shapes / index without re-reading the files every time.
PRIMARY_VOCAB: list[str] = load_vocabulary("primary")
SECONDARY_VOCAB: list[str] = load_vocabulary("moses")


_ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _coerce_probs(name: str, arr: _ArrayLike, n_rows: int, vocab_size: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise SubmissionError(
            f"{name} must be 2-D (got shape {a.shape}); expected ({n_rows}, {vocab_size})"
        )
    if a.shape != (n_rows, vocab_size):
        raise SubmissionError(
            f"{name} shape {a.shape} does not match expected ({n_rows}, {vocab_size})"
        )
    if not np.isfinite(a).all():
        raise SubmissionError(f"{name} contains NaN or infinite values")
    if (a < 0).any():
        raise SubmissionError(f"{name} contains negative values; expected probabilities")
    return a


def write_submission(
    path: Union[str, Path],
    indices: Union[Iterable[int], np.ndarray],
    primary_probs: _ArrayLike,
    secondary_probs: Optional[_ArrayLike] = None,
    *,
    primary_vocab: Optional[Sequence[str]] = None,
    secondary_vocab: Optional[Sequence[str]] = None,
    secondary_prefix: str = SECONDARY_VOCAB_PREFIX,
    normalize: bool = False,
    float_format: str = "{:.6f}",
) -> Path:
    """Write a competition submission CSV to ``path``.

    Parameters
    ----------
    path:
        Destination CSV path.
    indices:
        1-D sequence of row indices, length ``N``.
    primary_probs:
        ``(N, V1)`` array of probabilities over the primary vocabulary, in the
        same word order as :data:`PRIMARY_VOCAB`.
    secondary_probs:
        Optional ``(N, V2)`` array of probabilities over the secondary
        ("moses") vocabulary. If omitted, the submission omits the secondary
        columns entirely.
    primary_vocab, secondary_vocab:
        Override the bundled vocabularies (e.g. for testing).
    secondary_prefix:
        Prefix prepended to secondary-vocab column names.
    normalize:
        If ``True``, each row of each probability block is rescaled to sum to
        1. Defaults to ``False``; rows are passed through as-is and only sanity
        checks (non-negative, finite) are enforced.
    float_format:
        :func:`str.format` template applied to each probability value.

    Returns
    -------
    Path
        The absolute path the submission was written to.
    """
    primary_vocab = list(primary_vocab) if primary_vocab is not None else list(PRIMARY_VOCAB)
    secondary_vocab = (
        list(secondary_vocab) if secondary_vocab is not None else list(SECONDARY_VOCAB)
    )

    idx_arr = np.asarray(list(indices))
    if idx_arr.ndim != 1:
        raise SubmissionError(f"indices must be 1-D (got shape {idx_arr.shape})")
    n_rows = idx_arr.shape[0]

    primary = _coerce_probs("primary_probs", primary_probs, n_rows, len(primary_vocab))
    if normalize:
        s = primary.sum(axis=1, keepdims=True)
        if (s <= 0).any():
            raise SubmissionError("primary_probs has a row summing to <= 0; cannot normalize")
        primary = primary / s

    secondary: Optional[np.ndarray] = None
    if secondary_probs is not None:
        secondary = _coerce_probs(
            "secondary_probs", secondary_probs, n_rows, len(secondary_vocab)
        )
        if normalize:
            s = secondary.sum(axis=1, keepdims=True)
            if (s <= 0).any():
                raise SubmissionError(
                    "secondary_probs has a row summing to <= 0; cannot normalize"
                )
            secondary = secondary / s

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    header = ["index"] + list(primary_vocab)
    if secondary is not None:
        header += [f"{secondary_prefix}{w}" for w in secondary_vocab]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_rows):
            row = [int(idx_arr[i])] + [float_format.format(p) for p in primary[i]]
            if secondary is not None:
                row += [float_format.format(p) for p in secondary[i]]
            w.writerow(row)

    return out


def submit_to_kaggle(
    csv_path: Union[str, Path],
    competition: str,
    message: str = "",
    *,
    api_token: Optional[str] = None,
    kaggle_bin: Optional[str] = None,
    timeout: float = 180.0,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Upload ``csv_path`` to a Kaggle competition via the ``kaggle`` CLI.

    The function shells out to the official Kaggle CLI so that all of its
    authentication modes work transparently:

    * ``KAGGLE_API_TOKEN`` env var (new ``KGAT_…`` token format, CLI ≥ 2.0)
    * ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` env vars
    * ``~/.kaggle/kaggle.json`` or ``~/.kaggle/access_token``
    * ``kaggle auth login`` (OAuth)

    Parameters
    ----------
    csv_path:
        Path to the submission CSV.
    competition:
        Competition slug (e.g. ``"pnpl-internal-testing"``).
    message:
        Optional submission description shown on Kaggle.
    api_token:
        If provided, exported as ``KAGGLE_API_TOKEN`` for the subprocess only
        (never written to disk).
    kaggle_bin:
        Override the ``kaggle`` executable path. When ``None`` (default), the
        function invokes ``python -m kaggle`` using :data:`sys.executable` so
        the active interpreter's ``kaggle`` package is used — important on
        systems with multiple Python installs where ``shutil.which('kaggle')``
        could otherwise return a stale version from a different env.
    timeout:
        Seconds before the subprocess is killed.
    check:
        If ``True``, raise :class:`SubmissionError` on a non-zero exit.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the CLI call, with ``stdout`` / ``stderr`` captured.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise SubmissionError(f"Submission file not found: {csv_path}")

    if kaggle_bin:
        cmd_prefix = [kaggle_bin]
    else:
        # Use the kaggle that ships with the *current* interpreter, not whatever
        # `shutil.which('kaggle')` finds first on PATH (anaconda envs etc. often
        # shadow venv installs with an older, incompatible kaggle binary).
        try:
            import kaggle  # noqa: F401
        except ImportError as e:
            raise SubmissionError(
                "The 'kaggle' package is not installed in this interpreter "
                f"({sys.executable}). Install it with "
                "`pip install -U kaggle` (>= 2.0 for KAGGLE_API_TOKEN support)."
            ) from e
        # The kaggle package has no top-level __main__; the CLI entry point is
        # in kaggle.cli, so invoke that module directly.
        cmd_prefix = [sys.executable, "-m", "kaggle.cli"]

    env = os.environ.copy()
    if api_token:
        env["KAGGLE_API_TOKEN"] = api_token

    cmd = cmd_prefix + [
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        str(csv_path),
        "-m",
        message,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise SubmissionError(f"Failed to invoke kaggle CLI: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SubmissionError(f"kaggle submit timed out after {timeout}s") from e

    if check and proc.returncode != 0:
        raise SubmissionError(
            f"kaggle submit failed (exit {proc.returncode})\n"
            f"stdout: {proc.stdout.strip()}\n"
            f"stderr: {proc.stderr.strip()}"
        )

    return proc
