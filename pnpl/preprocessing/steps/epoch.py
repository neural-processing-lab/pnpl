"""
Epoch Step.

Creates epochs from continuous data based on events.
This is typically used for MegNIST-style datasets where the final
output should be epoched rather than continuous.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..pipeline import register_step
from .base import BaseStep

if TYPE_CHECKING:
    import mne


@register_step("epo")
@dataclass  
class Epoch(BaseStep):
    """
    Create epochs from continuous data.
    
    This step is typically the last in a pipeline for epoched datasets.
    
    Args:
        event_id: Event ID dictionary (e.g., {'digit/zero': 10, ...})
        tmin: Start time relative to event (default: -0.05)
        tmax: End time relative to event (default: 0.95)
        stim_channel: Stimulus channel name (default: 'STI101')
        min_duration: Minimum event duration (default: 0.005)
        baseline: Baseline correction interval (default: None)
    """
    
    step_name: str = "epo"
    event_id: Optional[Dict[str, int]] = None
    tmin: float = -0.05
    tmax: float = 0.95
    stim_channel: str = "STI101"
    min_duration: float = 0.005
    baseline: Optional[tuple] = None
    
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        """
        Note: This step modifies context to include epochs.
        The raw object is returned unchanged, but context['epochs'] contains
        the epoched data.
        """
        import mne
        import numpy as np
        
        # Find events
        events = mne.find_events(
            raw,
            stim_channel=self.stim_channel,
            min_duration=self.min_duration,
            mask=255,
            verbose=False,
        )
        
        # Use provided event_id or default MegNIST digits
        event_id = self.event_id
        if event_id is None:
            event_id = {
                'digit/zero': 10,
                'digit/one': 11,
                'digit/two': 12,
                'digit/three': 13,
                'digit/four': 14,
                'digit/five': 15,
                'digit/six': 16,
                'digit/seven': 17,
                'digit/eight': 18,
                'digit/nine': 19,
            }
        
        # Adjust event sample numbers if data was downsampled
        # This assumes original sfreq was 1000 Hz
        original_sfreq = context.get('original_sfreq', 1000)
        current_sfreq = raw.info['sfreq']
        
        if current_sfreq != original_sfreq:
            scaling_factor = current_sfreq / original_sfreq
            events[:, 0] = np.round(events[:, 0] * scaling_factor).astype(int)
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            verbose=False,
        )
        
        # Crop to exact number of samples
        epochs.crop(tmin=epochs.tmin, tmax=epochs.times[-2])
        epochs.pick('meg')
        
        # Store in context
        context['epochs'] = epochs
        context['events'] = events
        
        print(f"  Created {len(epochs)} epochs ({epochs.get_data().shape})")
        
        return raw

