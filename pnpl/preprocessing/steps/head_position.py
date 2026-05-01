"""
Head Position Loading Step.

Loads cached head position data from CSV files.
Note: This does NOT compute head positions (which is slow).
It loads pre-computed values from the derivatives/preproc/headpos/ directory.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("headpos")
@dataclass
class HeadPosition(BaseStep):
    """
    Load cached head position data.
    
    Head positions are loaded from CSV files in derivatives/preproc/headpos/.
    These should be pre-computed using mne.chpi.compute_chpi_locs and
    mne.chpi.compute_head_pos.
    
    The head positions are stored in context['head_pos'] for use by MaxwellFilter.
    """
    
    step_name: str = "headpos"
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        import mne
        
        bids_root = context.get('bids_root', '.')
        subject = context['subject']
        session = context['session']
        task = context['task']
        run = context['run']
        
        # Construct path to cached head position file
        headpos_dir = os.path.join(bids_root, "derivatives", "preproc", "headpos")
        headpos_file = os.path.join(
            headpos_dir,
            f"sub-{subject}_ses-{session}_task-{task}_run-{run}.csv"
        )
        
        if not os.path.exists(headpos_file):
            print(f"  Warning: Head position file not found: {headpos_file}")
            print(f"  Skipping head position correction")
            context['head_pos'] = None
            return raw
        
        # Load head positions using MNE's standard function
        head_pos = mne.chpi.read_head_pos(headpos_file)
        context['head_pos'] = head_pos
        
        print(f"  Loaded {len(head_pos)} head position samples")
        
        return raw

