# Make this subpackage composition-friendly so additional modules can be added
# or override existing ones without explicit registration.
try:
    from .._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

# Lazily expose convenience imports to avoid importing heavy dependencies at
# module import time while keeping `from pnpl.datasets import LibriBrainPhoneme`
# and similar imports available.
_PUBLIC_MAP = {
    # LibriBrain dataset entry point with an explicit task parameter
    "LibriBrain": ("pnpl.datasets.libribrain2025.dataset", "LibriBrain"),

    # Dataset-specific wrappers
    "LibriBrainPhoneme": ("pnpl.datasets.libribrain2025.compat", "LibriBrainPhoneme"),
    "LibriBrainSpeech": ("pnpl.datasets.libribrain2025.compat", "LibriBrainSpeech"),
    "LibriBrainWord": ("pnpl.datasets.libribrain2025.compat", "LibriBrainWord"),
    "LibriBrainSentence": ("pnpl.datasets.libribrain2025.sentence_dataset", "LibriBrainSentence"),
    
    # Utilities
    "GroupedDataset": ("pnpl.datasets.grouped_dataset", "GroupedDataset"),
}

__all__ = list(_PUBLIC_MAP.keys())

def __getattr__(name):  # pragma: no cover - import-time hook
    if name in _PUBLIC_MAP:
        modname, attr = _PUBLIC_MAP[name]
        from importlib import import_module
        mod = import_module(modname)
        return getattr(mod, attr)

    # Fallback to optional additional re-exports when available
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
