"""PNPL environment loading utilities.

PNPL reads configuration from environment variables (e.g. API keys).

This module provides a tiny, stdlib-only `.env` loader so users can just
create a `.env` file and run Python without manually `source`-ing it.

Behavior:
- Finds a `.env` by walking up from the current working directory.
- Loads `KEY=VALUE` pairs into `os.environ`.
- Does not override already-set environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _is_truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _find_dotenv(start_dir: Path, filename: str = ".env") -> Optional[Path]:
    """Search upward from start_dir for filename, returning the first match.

    We stop searching when we hit a likely project root marker, or when we hit
    the user's home directory. This avoids "global" dotenv files accidentally
    affecting unrelated scripts.
    """
    home = Path.home().resolve()
    allow_global = _is_truthy(os.getenv("PNPL_DOTENV_ALLOW_GLOBAL"))

    # Heuristic "project root" markers. If none exist, we still search upward
    # but will stop at $HOME (or filesystem root).
    stop_markers = {
        ".git",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
    }

    cur = start_dir
    for _ in range(100):  # safety cap
        candidate = cur / filename
        if candidate.is_file():
            # Avoid loading `.env` from $HOME or `/` unless explicitly allowed.
            try:
                parent = candidate.parent.resolve()
                if not allow_global and (parent == home or str(parent) == "/"):
                    return None
            except Exception:
                pass
            return candidate

        # If we are at a project root marker, don't keep walking upward.
        try:
            if any((cur / m).exists() for m in stop_markers):
                return None
        except Exception:
            pass

        # Avoid walking beyond $HOME by default.
        try:
            if cur.resolve() == home:
                return None
        except Exception:
            pass

        if cur.parent == cur:
            return None
        cur = cur.parent
    return None


def _parse_line(line: str) -> Optional[tuple[str, str]]:
    """Parse a single dotenv line into (key, value) or return None."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    if s.startswith("export "):
        s = s[len("export ") :].lstrip()

    if "=" not in s:
        return None

    key, val = s.split("=", 1)
    key = key.strip()
    if not key:
        return None

    val = val.strip()

    # Remove surrounding quotes.
    if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
        val = val[1:-1]

    return key, val


def load_dotenv(*, override: bool = False) -> Optional[Path]:
    """Load environment variables from a `.env` file.

    Resolution order:
    1) If `PNPL_DOTENV_PATH` is set, use that file.
    2) Else, walk up from `os.getcwd()` looking for `.env`.
    3) Else, walk up from the entry script directory (when running a .py file).

    If `PNPL_DISABLE_DOTENV` is truthy, this function is a no-op.

    Returns the loaded path (if any), otherwise None.
    """
    if _is_truthy(os.getenv("PNPL_DISABLE_DOTENV")):
        return None

    env_path = os.getenv("PNPL_DOTENV_PATH")
    if env_path:
        path = Path(env_path).expanduser()
    else:
        # 1) Prefer cwd-based lookup.
        path = _find_dotenv(Path.cwd())
        # 2) Fallback: when the user runs a script from another directory,
        # try locating `.env` relative to that entry script.
        if not path:
            try:
                argv0 = sys.argv[0] if sys.argv else ""
                p0 = Path(argv0).expanduser()
                if argv0 and p0.suffix == ".py" and p0.exists():
                    path = _find_dotenv(p0.resolve().parent)
            except Exception:
                path = None
    if not path or not path.is_file():
        return None

    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(raw)
            if not parsed:
                continue
            k, v = parsed
            if override:
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)
    except Exception:
        # Never fail import-time code paths due to dotenv issues.
        return None

    return path
