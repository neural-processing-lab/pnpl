"""
EpochedH5Mixin - Provides loading for epoched MEG data stored in H5 format.

This mixin handles H5 files with structure:
- data: (trials, channels, time) - Pre-epoched MEG data
- labels: (trials,) - Label for each trial
- times: (time,) - Time vector for each epoch
- channel_names: (channels,) - Channel names
- Metadata attributes: sensor_xyz, etc.

Used by MegNIST-style datasets where data is already epoched.
"""

import os
import h5py
import numpy as np
from typing import Optional, Union


class EpochedH5Mixin:
    """
    Mixin for loading epoched H5 MEG data.
    
    Classes using this mixin should have:
    - data_path: str - Base data directory
    """
    
    # Loaded data arrays
    _epoched_data: Optional[np.ndarray] = None
    _epoched_labels: Optional[np.ndarray] = None
    _epoched_times: Optional[np.ndarray] = None
    _channel_names: Optional[list] = None
    _sensor_xyz: Optional[np.ndarray] = None
    
    def get_epoched_h5_path(
        self,
        partition: str,
        preprocessing: Optional[str] = None,
    ) -> str:
        """
        Construct path to epoched H5 file.

        Args:
            partition: 'train', 'validation', or 'test'
            preprocessing: Optional preprocessing string

        Returns:
            Full path to H5 file
        """
        # Map partition names
        partition_map = {
            "train": "train",
            "training": "train",
            "val": "val",
            "validation": "val",
            "test": "test",
        }
        fname_base = partition_map.get(partition, partition)
        
        if preprocessing:
            fname = f"{fname_base}_proc-{preprocessing}.h5"
        else:
            fname = f"{fname_base}.h5"
        
        return os.path.join(self.data_path, "derivatives", "serialised", fname)
    
    def load_epoched_h5(
        self,
        h5_path: str,
        preload: bool = True,
    ) -> dict:
        """
        Load epoched data from H5 file.
        
        Args:
            h5_path: Path to H5 file
            preload: Whether to load data into memory
            
        Returns:
            Dict with 'data', 'labels', 'times', 'channel_names', 'sensor_xyz'
        """
        result = {}
        
        with h5py.File(h5_path, "r") as f:
            if preload:
                result['data'] = f['data'][:]
                result['labels'] = f['labels'][:]
            else:
                # Return file handle for lazy loading
                result['data'] = f['data']
                result['labels'] = f['labels']
            
            if 'times' in f:
                result['times'] = f['times'][:]
            
            if 'channel_names' in f:
                # Decode bytes to strings if needed
                names = f['channel_names'][:]
                if isinstance(names[0], bytes):
                    names = [n.decode('utf-8') for n in names]
                result['channel_names'] = list(names)
            
            if 'channel_types' in f:
                types = f['channel_types'][:]
                if isinstance(types[0], bytes):
                    types = [t.decode('utf-8') for t in types]
                result['channel_types'] = list(types)
            
            if 'sensor_xyz' in f:
                result['sensor_xyz'] = f['sensor_xyz'][:]
        
        return result
    
    def init_epoched_data(
        self,
        partition: str,
        preprocessing: Optional[str] = None,
        preload: bool = True,
    ) -> None:
        """
        Initialize epoched data from H5 file.

        Args:
            partition: Dataset partition ('train', 'validation', 'test')
            preprocessing: Optional preprocessing string
            preload: Whether to load all data into memory
        """
        h5_path = self.get_epoched_h5_path(partition, preprocessing)
        
        # Ensure file exists (download if needed)
        if hasattr(self, 'ensure_file'):
            h5_path = self.ensure_file(h5_path)
        
        data = self.load_epoched_h5(h5_path, preload)
        
        self._epoched_data = data['data']
        self._epoched_labels = data['labels']
        self._epoched_times = data.get('times')
        self._channel_names = data.get('channel_names')
        self._sensor_xyz = data.get('sensor_xyz')
    
    @property
    def data(self) -> np.ndarray:
        """Get the epoched data array (trials, channels, time)."""
        return self._epoched_data
    
    @property
    def labels(self) -> np.ndarray:
        """Get the labels array (trials,)."""
        return self._epoched_labels
    
    @property
    def times(self) -> Optional[np.ndarray]:
        """Get the time vector (time,)."""
        return self._epoched_times
    
    @property
    def n_trials(self) -> int:
        """Get number of trials."""
        if self._epoched_data is not None:
            return self._epoched_data.shape[0]
        return 0
    
    @property
    def n_channels(self) -> int:
        """Get number of channels."""
        if self._epoched_data is not None:
            return self._epoched_data.shape[1]
        return 0
    
    @property
    def n_times(self) -> int:
        """Get number of time points per epoch."""
        if self._epoched_data is not None:
            return self._epoched_data.shape[2]
        return 0
    
    def get_epoch(self, idx: int) -> np.ndarray:
        """
        Get a single epoch by index.
        
        Args:
            idx: Trial index
            
        Returns:
            (channels, time) array
        """
        return self._epoched_data[idx]
    
    def get_label(self, idx: int) -> Union[int, np.integer]:
        """
        Get label for a trial.
        
        Args:
            idx: Trial index
            
        Returns:
            Label value
        """
        return self._epoched_labels[idx]

