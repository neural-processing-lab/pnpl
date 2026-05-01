"""
Base task protocol and utilities.

Tasks define how samples are collected from datasets and how labels are extracted.
They follow a Protocol-based design for flexibility and type safety.
"""

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class TaskProtocol(Protocol):
    """
    Protocol defining what a Task must implement.
    
    Tasks are responsible for:
    1. Collecting samples from dataset metadata (e.g., events files)
    2. Transforming raw labels into the desired format
    3. Providing label metadata (classes, mappings)
    
    Example usage:
        @dataclass
        class MyTask:
            tmin: float = 0.0
            tmax: float = 0.5
            
            def collect_samples(self, dataset) -> list[tuple]:
                # Extract samples from dataset
                ...
            
            def get_label(self, sample: tuple) -> Any:
                # Transform sample tuple into label
                ...
            
            @property
            def label_info(self) -> dict:
                return {'classes': [...], 'label_to_id': {...}}
    """
    
    def collect_samples(self, dataset: "Dataset") -> list[tuple]:
        """
        Collect sample tuples from dataset metadata.
        
        This method is called during dataset initialization to gather
        all available samples based on the task requirements.
        
        Args:
            dataset: The dataset instance (provides access to data paths,
                    events files, run keys, etc.)
                    
        Returns:
            List of sample tuples. The format depends on the task:
            - Continuous data: (subject, session, task, run, onset, label, ...)
            - Epoched data: (trial_idx, label, ...)
        """
        ...
    
    def get_label(self, sample: tuple) -> Any:
        """
        Extract/transform label from a sample tuple.
        
        This method is called in __getitem__ to convert the raw sample
        tuple into the final label format.
        
        Args:
            sample: A sample tuple as returned by collect_samples
            
        Returns:
            The label in the desired format (int, array, etc.)
        """
        ...
    
    @property
    def label_info(self) -> dict:
        """
        Return label metadata.
        
        Returns:
            Dictionary containing:
            - 'classes': List of class names/values
            - 'label_to_id': Dict mapping label names to integer IDs
            - 'id_to_label': Dict mapping integer IDs to label names (optional)
            - 'n_classes': Number of classes (optional, derived from classes)
        """
        ...


class BaseTask:
    """
    Optional base class providing common task utilities.
    
    Tasks don't need to inherit from this class - they just need to
    implement the TaskProtocol. This class provides convenience methods.
    """
    
    _classes: list = None
    _label_to_id: dict = None
    
    @property
    def label_info(self) -> dict:
        """Default label_info implementation."""
        if self._classes is None:
            return {'classes': [], 'label_to_id': {}, 'n_classes': 0}
        
        if self._label_to_id is None:
            self._label_to_id = {c: i for i, c in enumerate(self._classes)}
        
        return {
            'classes': self._classes,
            'label_to_id': self._label_to_id,
            'id_to_label': {i: c for c, i in self._label_to_id.items()},
            'n_classes': len(self._classes),
        }
    
    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self._classes) if self._classes else 0
    
    @property
    def classes(self) -> list:
        """List of class names/values."""
        return self._classes or []

