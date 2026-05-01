"""
Preprocessing steps for MEG data.

Each step is a self-contained preprocessing operation that can be
composed into a Pipeline.
"""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

from .base import BaseStep
from .bad_channels import BadChannels
from .head_position import HeadPosition
from .maxwell_filter import MaxwellFilter
from .notch_filter import NotchFilter
from .bandpass_filter import BandpassFilter
from .downsample import Downsample
from .epoch import Epoch

__all__ = [
    "BaseStep",
    "BadChannels",
    "HeadPosition",
    "MaxwellFilter",
    "NotchFilter",
    "BandpassFilter",
    "Downsample",
    "Epoch",
]
