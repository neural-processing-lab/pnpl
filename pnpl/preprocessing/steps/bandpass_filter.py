"""
Bandpass Filter Step.

Applies a bandpass filter to isolate frequencies of interest.
"""

from dataclasses import dataclass
from typing import Dict, Any, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("bp")
@dataclass
class BandpassFilter(BaseStep):
    """
    Apply bandpass filter.
    
    Args:
        l_freq: Low cutoff frequency (default: 0.1 Hz)
        h_freq: High cutoff frequency (default: 125 Hz)
    """
    
    step_name: str = "bp"
    l_freq: float = 0.1
    h_freq: float = 125.0
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            picks='meg',
            verbose=False,
        )
        
        print(f"  Applied bandpass filter: {self.l_freq}-{self.h_freq} Hz")
        
        return raw

