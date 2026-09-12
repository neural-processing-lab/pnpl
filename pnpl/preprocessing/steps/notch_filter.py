"""
Notch Filter Step.

Removes line noise at specified frequencies (typically 50/60 Hz and harmonics).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("notch")
@dataclass
class NotchFilter(BaseStep):
    """
    Apply notch filter to remove line noise.
    
    Args:
        freqs: Frequencies to notch out (default: [50, 100] Hz for Europe)
        picks: Channels to filter, as accepted by :meth:`mne.io.Raw.notch_filter`
            (default: ``"meg"``; use ``"eeg"`` or ``"data"`` for EEG recordings)
    """
    
    step_name: str = "notch"
    freqs: List[float] = field(default_factory=lambda: [50.0, 100.0])
    picks: Any = "meg"
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        raw.notch_filter(
            freqs=self.freqs,
            picks=self.picks,
            verbose=False,
        )
        
        print(f"  Applied notch filter at {self.freqs} Hz")
        
        return raw

