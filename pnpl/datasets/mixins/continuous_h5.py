"""
ContinuousH5Mixin - Provides loading for continuous MEG data stored in H5 format.

This mixin handles H5 files with structure:
- data: (channels, time_samples) - Continuous MEG data
- times: (time_samples,) - Time vector
- Metadata attributes: sample_frequency, channel_names, etc.

Used by LibriBrain-style datasets where samples are extracted as time windows.
"""

import os
import h5py
import numpy as np
from typing import Optional


class ContinuousH5Mixin:
    """
    Mixin for loading continuous H5 MEG data.
    
    Classes using this mixin should have:
    - data_path: str - Base data directory
    - preprocessing: str - Preprocessing string for filename
    - sfreq: float - Sampling frequency
    - tmin, tmax: float - Time window parameters
    """
    
    # Cache for open H5 file handles (keep File objects alive; Dataset objects
    # alone can become invalid if the File is garbage-collected).
    _open_h5_files: dict = None
    
    def init_continuous_h5(self) -> None:
        """Initialize the H5 data cache."""
        self._open_h5_files = {}
    
    def get_h5_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preprocessing: Optional[str] = None,
    ) -> str:
        """
        Construct path to H5 file.
        
        Args:
            subject, session, task, run: Identifiers
            preprocessing: Override preprocessing string (uses self.preprocessing if None)
            
        Returns:
            Full path to H5 file
        """
        proc = preprocessing or getattr(self, 'preprocessing', None)
        
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
        if proc is not None:
            fname += f"_proc-{proc}"
        fname += "_meg.h5"
        
        return os.path.join(self.data_path, task, "derivatives", "serialised", fname)
    
    def get_events_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> str:
        """
        Construct path to events TSV file.
        
        Args:
            subject, session, task, run: Identifiers
            
        Returns:
            Full path to events.tsv file
        """
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        return os.path.join(self.data_path, task, "derivatives", "events", fname)
    
    def get_sfreq_from_h5(self, h5_path: str) -> float:
        """
        Get sampling frequency from H5 file.
        
        Args:
            h5_path: Path to H5 file
            
        Returns:
            Sampling frequency in Hz
        """
        with h5py.File(h5_path, "r") as h5_file:
            return h5_file.attrs["sample_frequency"]
    
    def get_h5_dataset(self, run_key: tuple) -> h5py.Dataset:
        """
        Get (cached) H5 dataset for a run.
        
        Args:
            run_key: (subject, session, task, run) tuple
            
        Returns:
            h5py Dataset object for the 'data' array
        """
        if self._open_h5_files is None:
            self._open_h5_files = {}

        if run_key not in self._open_h5_files:
            subject, session, task, run = run_key
            h5_path = self.get_h5_path(subject, session, task, run)
            
            # Ensure file exists (download if needed)
            if hasattr(self, 'ensure_file'):
                h5_path = self.ensure_file(h5_path)
            
            self._open_h5_files[run_key] = h5py.File(h5_path, "r")

        return self._open_h5_files[run_key]["data"]
    
    def load_continuous_window(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        onset: float,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load a time window from continuous H5 data.
        
        Args:
            subject, session, task, run: Identifiers
            onset: Event onset time in seconds
            tmin: Start time relative to onset (uses self.tmin if None)
            tmax: End time relative to onset (uses self.tmax if None)
            
        Returns:
            (channels, time) array
        """
        tmin = tmin if tmin is not None else self.tmin
        tmax = tmax if tmax is not None else self.tmax
        sfreq = self.sfreq
        points_per_sample = int((tmax - tmin) * sfreq)
        
        run_key = (subject, session, task, run)
        h5_dataset = self.get_h5_dataset(run_key)
        
        start = max(0, int((onset + tmin) * sfreq))
        end = start + points_per_sample
        
        return h5_dataset[:, start:end]
    
    def load_continuous_window_from_sample(self, sample: tuple) -> np.ndarray:
        """
        Load time window from a sample tuple.
        
        Args:
            sample: Tuple containing at least (subject, session, task, run, onset, ...)
            
        Returns:
            (channels, time) array
        """
        subject, session, task, run, onset = sample[:5]
        return self.load_continuous_window(subject, session, task, run, onset)
    
    def close_h5_files(self) -> None:
        """Close all open H5 file handles."""
        if self._open_h5_files:
            for h5_file in self._open_h5_files.values():
                try:
                    h5_file.close()
                except Exception:
                    pass
            self._open_h5_files.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_h5_files()
