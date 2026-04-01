"""
Base class for preprocessing steps.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import mne


class BaseStep(ABC):
    """
    Abstract base class for preprocessing steps.
    
    Each step must implement:
    - step_name: Short name used in filename strings (e.g., 'bads', 'sss')
    - apply(): Method that transforms the raw data
    """
    
    step_name: str = "base"
    
    @abstractmethod
    def apply(self, raw: "mne.io.Raw", context: Dict[str, Any]) -> "mne.io.Raw":
        """
        Apply this preprocessing step to raw data.
        
        Args:
            raw: MNE Raw object to preprocess
            context: Dictionary with shared context between steps:
                - subject, session, task, run: BIDS identifiers
                - bids_root: Path to BIDS root
                - head_pos: Head position array (set by HeadPosition step)
                - bad_channels: Dict with 'noisy' and 'flat' lists
                
        Returns:
            Preprocessed MNE Raw object
        """
        pass

