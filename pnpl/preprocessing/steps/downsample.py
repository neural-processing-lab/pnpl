"""
Downsample Step.

Reduces the sampling rate of the data.
"""

from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("ds")
@dataclass
class Downsample(BaseStep):
    """
    Downsample the data.
    
    Args:
        sfreq: Target sampling frequency (default: 250 Hz)
    """
    
    step_name: str = "ds"
    sfreq: float = 250.0
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        original_sfreq = raw.info['sfreq']
        
        if original_sfreq > self.sfreq:
            raw.resample(sfreq=self.sfreq, verbose=False)
            print(f"  Downsampled from {original_sfreq} Hz to {self.sfreq} Hz")
        else:
            print(f"  Already at {original_sfreq} Hz (target: {self.sfreq} Hz)")
        
        return raw

