"""
Mixins for dataset functionality.

These mixins provide reusable functionality that can be composed into dataset classes.
"""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

from .download import HFDownloadMixin
from .ohana_download import OhanaDownloadMixin
from .osf_download import OSFDownloadMixin
from .standardization import StandardizationMixin
from .continuous_h5 import ContinuousH5Mixin
from .epoched_h5 import EpochedH5Mixin
from .bids import BIDSMixin

__all__ = [
    "HFDownloadMixin",
    "OhanaDownloadMixin",
    "OSFDownloadMixin",
    "StandardizationMixin",
    "ContinuousH5Mixin",
    "EpochedH5Mixin",
    "BIDSMixin",
]
