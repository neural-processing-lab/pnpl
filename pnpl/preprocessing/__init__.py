"""
Preprocessing module for MEG data.

Provides a modular pipeline for preprocessing raw MEG data:
- BadChannels: Detect and mark bad channels
- HeadPosition: Load cached head position data
- MaxwellFilter: Apply SSS/Maxwell filtering
- NotchFilter: Remove line noise
- BandpassFilter: Apply bandpass filtering
- Downsample: Reduce sampling rate
- Epoch: Create epochs from continuous data (for MegNIST-style datasets)
"""

try:
    from .._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

from .pipeline import Pipeline
from .steps import (
    BaseStep,
    BadChannels,
    HeadPosition,
    MaxwellFilter,
    NotchFilter,
    BandpassFilter,
    Downsample,
    Epoch,
)
from .serialization import fif_to_h5, epochs_to_h5

__all__ = [
    "Pipeline",
    "BaseStep",
    "BadChannels",
    "HeadPosition",
    "MaxwellFilter",
    "NotchFilter",
    "BandpassFilter",
    "Downsample",
    "Epoch",
    "fif_to_h5",
    "epochs_to_h5",
]
