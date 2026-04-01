"""
StandardizationMixin - Provides z-score normalization and clipping functionality.

This mixin handles:
- Computing channel means and standard deviations across runs
- Z-score normalization of MEG data
- Value clipping to handle outliers
"""

import numpy as np
from typing import Optional


class StandardizationMixin:
    """
    Mixin providing standardization functionality for MEG data.
    
    Classes using this mixin should have:
    - points_per_sample: int - Number of time points per sample
    - run_keys: list - List of run keys for computing stats
    - A method to load H5 data for computing statistics
    """
    
    # Instance attributes set during init or by _calculate_standardization_params
    channel_means: Optional[np.ndarray] = None
    channel_stds: Optional[np.ndarray] = None
    broadcasted_means: Optional[np.ndarray] = None
    broadcasted_stds: Optional[np.ndarray] = None
    
    def setup_standardization(
        self,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set up standardization parameters.
        
        Args:
            standardize: Whether to apply z-score normalization
            clipping_boundary: If set, clip values to [-boundary, boundary]
            channel_means: Pre-computed channel means (optional)
            channel_stds: Pre-computed channel stds (optional)
        """
        self.standardize_enabled = standardize
        self.clipping_boundary = clipping_boundary
        
        # Convert lists to arrays if needed
        if isinstance(channel_means, list):
            channel_means = np.array(channel_means)
        if isinstance(channel_stds, list):
            channel_stds = np.array(channel_stds)
        
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        
        # Pre-compute broadcasted arrays if we have the values
        if channel_means is not None and channel_stds is not None:
            self._broadcast_stats()
    
    def _broadcast_stats(self) -> None:
        """Pre-broadcast means and stds for efficient vectorized operations."""
        if hasattr(self, 'points_per_sample') and self.channel_means is not None:
            self.broadcasted_means = np.tile(
                self.channel_means, (self.points_per_sample, 1)
            ).T
            self.broadcasted_stds = np.tile(
                self.channel_stds, (self.points_per_sample, 1)
            ).T
    
    def calculate_standardization_params(self, h5_data_loader) -> None:
        """
        Calculate channel means and stds across all runs.
        
        Args:
            h5_data_loader: Callable that takes run_key and returns h5py dataset
        """
        n_samples = []
        means = []
        stds = []
        
        for run_key in self.run_keys:
            hdf_dataset = h5_data_loader(run_key)

            # Capture n_samples before any optional caching logic.
            # (Some caching approaches may open the file in r+ separately.)
            n_samples.append(hdf_dataset.shape[1])
            
            # Check for cached stats in H5 attributes
            if "channel_means" in hdf_dataset.attrs and "channel_stds" in hdf_dataset.attrs:
                channel_means = hdf_dataset.attrs["channel_means"]
                channel_stds = hdf_dataset.attrs["channel_stds"]
            else:
                # Compute stats from data
                data = hdf_dataset[:, :]
                channel_means = np.mean(data, axis=1)
                channel_stds = np.std(data, axis=1)
                
                # Try to cache the computed stats
                try:
                    import h5py
                    with h5py.File(hdf_dataset.file.filename, "r+") as f:
                        f["data"].attrs["channel_means"] = channel_means
                        f["data"].attrs["channel_stds"] = channel_stds
                    print(f"Cached stats for: {run_key}")
                except Exception:
                    pass  # Read-only file or other issue
            
            means.append(channel_means)
            stds.append(channel_stds)
        
        # Accumulate stats across runs
        means = np.array(means)
        stds = np.array(stds)
        n_samples = np.array(n_samples)
        
        self.channel_stds, self.channel_means = self._accumulate_stds(means, stds, n_samples)
        self._broadcast_stats()
    
    @staticmethod
    def _accumulate_stds(
        ch_means: np.ndarray,
        ch_stds: np.ndarray,
        n_samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Accumulate standard deviations across multiple groups using parallel algorithm.
        
        Args:
            ch_means: (n_groups, n_channels) - Mean per channel per group
            ch_stds: (n_groups, n_channels) - Std per channel per group
            n_samples: (n_groups,) - Number of samples per group
            
        Returns:
            (accumulated_stds, accumulated_means)
        """
        vars = np.array(ch_stds) ** 2
        means_total = np.average(ch_means, axis=0, weights=n_samples)
        
        sum_of_squares_within = np.sum(
            vars * np.tile(n_samples, (vars.shape[1], 1)).T, axis=0
        )
        sum_of_squares_between = np.sum(
            (ch_means - np.tile(means_total, (ch_means.shape[0], 1))) ** 2 *
            np.tile(n_samples, (ch_means.shape[1], 1)).T,
            axis=0
        )
        sum_of_squares_total = sum_of_squares_within + sum_of_squares_between
        
        return np.sqrt(sum_of_squares_total / np.sum(n_samples)), means_total
    
    def standardize(self, data: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization and optional clipping to data.
        
        Args:
            data: (channels, time) array
            
        Returns:
            Standardized (and optionally clipped) data
        """
        if self.standardize_enabled and self.broadcasted_means is not None:
            # Handle edge case where sample is smaller than expected
            if data.shape[1] < self.broadcasted_means.shape[1]:
                means = self.broadcasted_means[:, :data.shape[1]]
                stds = self.broadcasted_stds[:, :data.shape[1]]
            else:
                means = self.broadcasted_means
                stds = self.broadcasted_stds
            
            data = (data - means) / stds
        
        if self.clipping_boundary is not None:
            data = np.clip(data, -self.clipping_boundary, self.clipping_boundary)
        
        return data
    
    def clip_sample(self, sample: np.ndarray, boundary: float) -> np.ndarray:
        """Clip sample values to [-boundary, boundary]."""
        return np.clip(sample, -boundary, boundary)
