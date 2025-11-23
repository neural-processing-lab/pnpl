# Make this subpackage a namespace so the internal overlay can add
# additional subpackages/modules alongside the public ones.
try:  # pkgutil-style namespace for subpackage
    import pkgutil as _pkgutil  # stdlib
    __path__ = _pkgutil.extend_path(__path__, __name__)
except Exception:
    pass

# Lazily expose public convenience imports to avoid importing heavy
# dependencies at module import time. Keep backwards compatibility with
# `from pnpl.datasets import LibriBrainPhoneme`, etc.
_PUBLIC_MAP = {
    "LibriBrainPhoneme": (
        "pnpl.datasets.libribrain2025.phoneme_dataset", "LibriBrainPhoneme"
    ),
    "LibriBrainSpeech": (
        "pnpl.datasets.libribrain2025.speech_dataset", "LibriBrainSpeech"
    ),
    "LibriBrainSpeech": (
        "pnpl.datasets.libribrain2025.word_dataset", "LibriBrainWord"
    ),
    "GroupedDataset": ("pnpl.datasets.grouped_dataset", "GroupedDataset"),
    "LibriBrainCompetitionHoldout": (
        "pnpl.datasets.libribrain2025.competition_holdout_dataset",
        "LibriBrainCompetitionHoldout",
    ),
}

__all__ = list(_PUBLIC_MAP.keys())

def __getattr__(name):  # pragma: no cover - import-time hook
    if name in _PUBLIC_MAP:
        modname, attr = _PUBLIC_MAP[name]
        from importlib import import_module
        mod = import_module(modname)
        return getattr(mod, attr)

    # Fallback to optional overlay re-exports when available
    try:
        from importlib import import_module
        priv = import_module("pnpl.datasets._private_exports")
        return getattr(priv, name)
    except Exception as _e:
        raise AttributeError(name) from _e

def __dir__():  # pragma: no cover - import-time hook
    names = set(__all__)
    try:
        from importlib import import_module
        priv = import_module("pnpl.datasets._private_exports")
        names.update(n for n in dir(priv) if not n.startswith("_"))
    except Exception:
        pass
    return sorted(names)

# If an internal overlay is installed, it can publish additional public
# symbols for convenient imports: `from pnpl.datasets import X`.
# Importing this optional module is safe if the overlay is not present.
try:  # pragma: no cover - optional extension point
    from ._private_exports import *  # noqa: F401,F403
except Exception:
    # No internal overlay installed (or it failed to import); ignore.
    pass
