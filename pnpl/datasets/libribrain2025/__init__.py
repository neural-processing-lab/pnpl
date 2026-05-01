"""
LibriBrain dataset module.

Provides the main LibriBrain class plus compatibility wrappers without forcing
heavy imports at package import time.
"""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

_PUBLIC_MAP = {
    "LibriBrain": ("pnpl.datasets.libribrain2025.dataset", "LibriBrain"),
    "LibriBrainSpeech": ("pnpl.datasets.libribrain2025.compat", "LibriBrainSpeech"),
    "LibriBrainPhoneme": ("pnpl.datasets.libribrain2025.compat", "LibriBrainPhoneme"),
    "LibriBrainWord": ("pnpl.datasets.libribrain2025.compat", "LibriBrainWord"),
    "LibriBrainSentence": (
        "pnpl.datasets.libribrain2025.sentence_dataset",
        "LibriBrainSentence",
    ),
}

__all__ = list(_PUBLIC_MAP.keys())


def __getattr__(name):
    if name not in _PUBLIC_MAP:
        raise AttributeError(name)

    modname, attr = _PUBLIC_MAP[name]
    from importlib import import_module

    module = import_module(modname)
    return getattr(module, attr)


def __dir__():
    return sorted(__all__)
