"""HDF5 dataset helpers for ``pnpl.datasets.hdf5``."""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass
