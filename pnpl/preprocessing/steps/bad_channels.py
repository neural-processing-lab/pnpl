"""
Bad Channels Detection Step.

Detects noisy and flat channels using Maxwell filter-based detection.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("bads")
@dataclass
class BadChannels(BaseStep):
    """
    Detect and mark bad channels.
    
    Uses MNE's find_bad_channels_maxwell to detect noisy and flat channels.
    Bad channels are stored in context for use by MaxwellFilter.
    
    Args:
        h_freq: High frequency cutoff for detection (default: 40 Hz)
        calibration: Path to calibration file (auto-detected if None)
        cross_talk: Path to cross-talk file (auto-detected if None)
    """
    
    step_name: str = "bads"
    h_freq: float = 40.0
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
        
        # Initialize bads
        raw.info['bads'] = []
        
        # Detect bad channels
        noisy, flat = mne.preprocessing.find_bad_channels_maxwell(
            raw,
            calibration=cal_file,
            cross_talk=ct_file,
            h_freq=self.h_freq,
            verbose=False,
        )
        
        # Store in raw and context
        raw.info['bads'] = noisy + flat
        context['bad_channels'] = {
            'noisy': noisy,
            'flat': flat,
            'all': noisy + flat,
        }
        
        print(f"  Found {len(noisy)} noisy, {len(flat)} flat channels")
        
        return raw

