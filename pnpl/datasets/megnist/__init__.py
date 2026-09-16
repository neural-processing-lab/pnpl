"""MegNIST dataset module."""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

_PUBLIC_MAP = {
    "MegNIST": ("pnpl.datasets.megnist.dataset", "MegNIST"),
}

__all__ = list(_PUBLIC_MAP.keys())


def __getattr__(name):
    if name not in _PUBLIC_MAP:
        raise AttributeError(name)

    from importlib import import_module

    modname, attr = _PUBLIC_MAP[name]
    module = import_module(modname)
    return getattr(module, attr)


def __dir__():
    return sorted(__all__)