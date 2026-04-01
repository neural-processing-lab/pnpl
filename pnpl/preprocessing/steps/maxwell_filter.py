"""
Maxwell Filter (SSS) Step.

Applies Signal Space Separation to the data, optionally with head position
correction if head positions are available in the context.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("sss")
@dataclass
class MaxwellFilter(BaseStep):
    """
    Apply Maxwell filter (Signal Space Separation).
    
    This step:
    - Separates signals from inside and outside the head
    - Interpolates bad channels
    - Optionally applies head position correction
    
    Args:
        origin: Origin for spherical harmonics ('auto' or (x, y, z))
        destination: Destination for head position alignment (default: (0, 0, 0.04))
        calibration: Path to calibration file (auto-detected if None)
        cross_talk: Path to cross-talk file (auto-detected if None)
    """
    
    step_name: str = "sss"
    origin: str = "auto"
    destination: Tuple[float, float, float] = (0, 0, 0.04)
    calibration: Optional[str] = None
    cross_talk: Optional[str] = None
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        import mne
        
        # Get calibration files
        cal_file = self.calibration
        ct_file = self.cross_talk
        
        if cal_file is None or ct_file is None:
            bids_root = context.get('bids_root', '.')
            neo_dir = os.path.join(bids_root, "derivatives", "neo")
            if cal_file is None:
                cal_file = os.path.join(neo_dir, "sss_cal.dat")
            if ct_file is None:
                ct_file = os.path.join(neo_dir, "ct_sparse.fif")
        
        # Get head positions from context (set by HeadPosition step)
        head_pos = context.get('head_pos')
        
        # Apply Maxwell filter
        raw = mne.preprocessing.maxwell_filter(
            raw,
            origin=self.origin,
            coord_frame='head',
            destination=self.destination,
            calibration=cal_file,
            cross_talk=ct_file,
            head_pos=head_pos,
            verbose=False,
        )
        
        if head_pos is not None:
            print(f"  Applied Maxwell filter with head position correction")
        else:
            print(f"  Applied Maxwell filter (no head position correction)")
        
        return raw

