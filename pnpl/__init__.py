"""
PNPL - Python Neural Processing Lab.

A library for loading and processing MEG brain datasets.

Example usage:

    from pnpl.datasets import LibriBrain
    from pnpl.tasks import SpeechDetection
    
    # LibriBrain with speech detection
    train = LibriBrain(
        data_path="./data",
        task=SpeechDetection(tmin=0, tmax=0.5),
        partition="train",
    )
    
    from pnpl.datasets import LibriBrainSpeech
    speech = LibriBrainSpeech(data_path="./data", partition="train")
"""

__version__ = "0.0.8"

# Load environment variables from a local `.env` (if present). This keeps
# configuration ergonomic for users while remaining lightweight (stdlib-only).
try:  # pragma: no cover - import-time best effort
    from ._env import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

# 1) Make `pnpl` a composition-friendly namespace so multiple distributions can
# contribute and later-installed modules can override earlier ones when needed.
try:
    from ._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    # If anything goes wrong, keep default package path.
    pass

# Public submodules
__all__ = ["datasets", "tasks", "preprocessing"]

# 2) Optional lazy access to dataset exports at the top-level. We avoid
# importing subpackages eagerly to keep `import pnpl` lightweight.
def __getattr__(name):  # pragma: no cover - import-time hook
    # Try public submodules first
    if name in ("datasets", "tasks", "preprocessing"):
        from importlib import import_module
        return import_module(f"pnpl.{name}")

    # Dataset convenience imports are exposed through pnpl.datasets.
    try:
        from importlib import import_module

        datasets = import_module("pnpl.datasets")
        return getattr(datasets, name)
    except Exception as _e:
        raise AttributeError(name) from _e

def __dir__():  # pragma: no cover - import-time hook
    standard = list(__all__)
    try:
        from importlib import import_module

        mod = import_module("pnpl.datasets")
        return sorted(set(standard + [
            n for n in dir(mod) if not n.startswith("_")
        ]))
    except Exception:
        return sorted(standard)
