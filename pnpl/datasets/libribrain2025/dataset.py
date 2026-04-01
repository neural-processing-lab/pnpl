"""
LibriBrain Dataset - Main dataset class for continuous MEG data.

This dataset loads preprocessed MEG data from H5 files and supports
various tasks (speech detection, phoneme classification, word detection).
"""

import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Union, Dict, Any

from ..mixins import HFDownloadMixin, StandardizationMixin, ContinuousH5Mixin, BIDSMixin
from ..utils import check_include_and_exclude_ids, include_exclude_ids
from .constants import RUN_KEYS, VALIDATION_RUN_KEYS, TEST_RUN_KEYS


class LibriBrain(
    HFDownloadMixin,
    StandardizationMixin,
    ContinuousH5Mixin,
    BIDSMixin,
    Dataset
):
    """
    LibriBrain MEG dataset for speech and language tasks.
    
    This dataset provides continuous MEG recordings of a participant listening
    to audiobooks. Supports multiple tasks including speech detection,
    phoneme classification, and word detection.
    
    Args:
        data_path: Path to store/load the dataset
        task: Task object defining sample collection and labeling
        partition: Shortcut for train/validation/test split
        preprocessing: Preprocessing string (e.g., 'bads+headpos+sss+notch+bp+ds')
        preprocessing_config: Optional dict of preprocessing step configurations.
            Overrides defaults and JSON config. Format: {"step_name": {"param": value}}
        include_run_keys: Specific runs to include
        exclude_run_keys: Specific runs to exclude
        exclude_tasks: Task names to exclude (e.g., ['Sherlock1'])
        standardize: Whether to z-score normalize data
        clipping_boundary: Clip values to [-boundary, boundary]
        channel_means: Pre-computed channel means
        channel_stds: Pre-computed channel stds
        include_info: Include metadata dict in samples
        preload_files: Eagerly download files on init
        download: Enable downloading from HuggingFace
        
    Example:
        >>> from pnpl.datasets import LibriBrain
        >>> from pnpl.tasks import SpeechDetection
        >>> dataset = LibriBrain(
        ...     data_path="./data",
        ...     task=SpeechDetection(tmin=0, tmax=0.5),
        ...     partition="train",
        ... )
    """
    
    # HuggingFace repository configuration
    HUGGINGFACE_REPO = "pnpl/LibriBrain"
    HUGGINGFACE_FALLBACK_REPOS = ["pnpl/LibriBrain-Competition-2025"]
    
    def __init__(
        self,
        data_path: str,
        task,  # TaskProtocol
        partition: Optional[str] = None,
        preprocessing: str = "bads+headpos+sss+notch+bp+ds",
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
        include_run_keys: list = None,
        exclude_run_keys: list = None,
        exclude_tasks: list = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = True,
        download: bool = True,
    ):
        os.makedirs(data_path, exist_ok=True)
        
        # Store configuration
        self.data_path = data_path
        self.task = task
        self.partition = partition
        self.preprocessing = preprocessing
        self.preprocessing_config = preprocessing_config or {}
        self.include_info = include_info
        self.download = download
        
        # Initialize defaults
        include_run_keys = include_run_keys or []
        exclude_run_keys = exclude_run_keys or []
        exclude_tasks = exclude_tasks or []
        
        # Get time window from task
        self.tmin = getattr(task, 'tmin', 0.0)
        self.tmax = getattr(task, 'tmax', 0.5)
        
        # Handle partition shortcuts
        if partition is not None:
            if include_run_keys or exclude_run_keys or exclude_tasks:
                raise ValueError(
                    "partition is a shortcut - include_run_keys, exclude_run_keys, "
                    "and exclude_tasks must be empty when partition is specified"
                )
            if partition == "train":
                exclude_run_keys = list(VALIDATION_RUN_KEYS) + list(TEST_RUN_KEYS)
            elif partition == "validation":
                include_run_keys = list(VALIDATION_RUN_KEYS)
            elif partition == "test":
                include_run_keys = list(TEST_RUN_KEYS)
            else:
                raise ValueError(f"Invalid partition: {partition}. Must be train/validation/test")
        
        # Convert to tuples and validate
        include_run_keys = [tuple(rk) for rk in include_run_keys]
        exclude_run_keys = [tuple(rk) for rk in exclude_run_keys]
        check_include_and_exclude_ids(include_run_keys, exclude_run_keys, RUN_KEYS)
        
        # Compute intended run keys
        intended_run_keys = include_exclude_ids(include_run_keys, exclude_run_keys, RUN_KEYS)
        self.intended_run_keys = [rk for rk in intended_run_keys if rk[2] not in exclude_tasks]
        
        if not self.intended_run_keys:
            raise ValueError("No run keys match the specified configuration")
        
        # Initialize H5 caching
        self.init_continuous_h5()
        
        # Prefetch files if requested
        if preload_files and download:
            self._prefetch_dataset_files()
        
        # Get sampling frequency from first file
        first_key = self.intended_run_keys[0]
        h5_path = self.get_h5_path(*first_key)
        if download:
            h5_path = self.ensure_file(h5_path)
        self.sfreq = self.get_sfreq_from_h5(h5_path)
        self.points_per_sample = int((self.tmax - self.tmin) * self.sfreq)
        
        # Collect samples using the task
        self.samples = []
        self.run_keys = []
        run_keys_missing = []
        
        for run_key in self.intended_run_keys:
            try:
                # Verify H5 file exists
                h5_path = self.get_h5_path(*run_key)
                if download:
                    self.ensure_file(h5_path)
                elif not os.path.exists(h5_path):
                    raise FileNotFoundError(h5_path)
                
                self.run_keys.append(run_key)
            except FileNotFoundError:
                run_keys_missing.append(run_key)
                warnings.warn(f"File not found for run key {run_key}. Skipping")
        
        if run_keys_missing:
            warnings.warn(f"Run keys not found: {run_keys_missing}")
        
        if not self.run_keys:
            raise ValueError("No valid run keys found")
        
        # Collect samples via task
        self.samples = self.task.collect_samples(self)
        
        if not self.samples:
            raise ValueError("No samples found for the specified configuration")
        
        # Setup standardization
        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )
        
        # Calculate standardization params if not provided
        if standardize and channel_means is None and channel_stds is None:
            self._calculate_standardization_params()
    
    def _prefetch_dataset_files(self):
        """Prefetch all H5 and events files."""
        file_paths = []
        for run_key in self.intended_run_keys:
            file_paths.append(self.get_h5_path(*run_key))
            file_paths.append(self.get_events_path(*run_key))
        self.prefetch_files(file_paths)
    
    def _calculate_standardization_params(self):
        """Calculate channel means and stds across all runs."""
        def h5_loader(run_key):
            return self.get_h5_dataset(run_key)
        self.calculate_standardization_params(h5_loader)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx]
        
        # Load data from H5
        data = self.load_continuous_window_from_sample(sample)
        
        # Apply standardization
        data = self.standardize(data)
        
        # Get label from task
        label = self.task.get_label(sample)
        
        # Convert to tensors
        data = torch.tensor(data, dtype=torch.float32)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        elif isinstance(label, int):
            label = torch.tensor(label)
        
        # Build info dict if requested
        if self.include_info:
            info = {
                "dataset": "libribrain2025",
                "subject": sample[0],
                "session": sample[1],
                "task": sample[2],
                "run": sample[3],
                "onset": torch.tensor(sample[4], dtype=torch.float32),
            }
            return data, label, info
        
        return data, label
    
    @property
    def label_info(self):
        """Get label information from the task."""
        return self.task.label_info
    
    @property
    def n_channels(self):
        """Number of MEG channels (306 for Elekta/MEGIN)."""
        return 306
    
    @property
    def n_times(self):
        """Number of time points per sample."""
        return self.points_per_sample
