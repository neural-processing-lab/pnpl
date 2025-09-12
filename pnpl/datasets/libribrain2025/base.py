import os
import numpy as np
import pandas as pd
import h5py
import torch
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor
from pnpl.datasets.libribrain2025.constants import RUN_KEYS, PHONATION_BY_PHONEME
from pnpl.datasets.utils import check_include_and_exclude_ids, include_exclude_ids
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError, GatedRepoError
from requests.exceptions import ConnectionError, Timeout, HTTPError
from pnpl.datasets.libribrain2025.constants import VALIDATION_RUN_KEYS, TEST_RUN_KEYS


class LibriBrainBase(Dataset):
    # Adjust max_workers as needed.
    _executor = ThreadPoolExecutor(max_workers=4)
    _download_futures = {}
    _lock = threading.Lock()

    def __init__(
            self,
            data_path: str,
            partition: str | None = None,
            preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
            tmin: float = 0.0,
            tmax: float = 0.5,
            include_run_keys: list[str] = [],
            exclude_run_keys: list[str] = [],
            exclude_tasks: list[str] = [],
            standardize: bool = True,
            clipping_boundary: float | None = 10,
            channel_means: np.ndarray | None = None,
            channel_stds: np.ndarray | None = None,
            include_info: bool = False,
            preload_files: bool = True,
            preload_event_file: bool = True,
            download: bool = True
    ):
        """
        Base class for LibriBrain datasets.

        See the dataset on HuggingFace: https://huggingface.co/datasets/pnpl/LibriBrain

        Args:
            data_path: Path where you wish to store the dataset. The local dataset structure 
                      will follow the same BIDS-like structure as the HuggingFace repo:
                      ```
                      data_path/
                      ├── {task}/                    # e.g., "Sherlock1"
                      │   └── derivatives/
                      │       ├── serialised/       # MEG data files
                      │       │   └── sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5
                      │       └── events/            # Event files  
                      │           └── sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv
                      ```
            partition: Convenient shortcut to specify train/validation/test split. Use "train", 
                      "validation", or "test". When specified, include_run_keys, exclude_run_keys, 
                      and exclude_tasks must be empty. If None, you can manually specify run keys.
            preprocessing_str: By default, we expect files to be named with the full preprocessing 
                             string "bads+headpos+sss+notch+bp+ds". Override this if you want different 
                             preprocessing. The full string means:
                             - bads: Removal of bad channels
                             - headpos: Adjusted signal to compensate for head movement  
                             - sss: Signal-space separation to isolate signals from inside vs outside head
                             - notch: Notch filtering to eliminate specific frequency noise (e.g., electrical grid)
                             - bp: Bandpass filtering to isolate frequency range of interest
                             - ds: Downsampling to reduce sampling rate for easier handling
            tmin: Start time of the sample in seconds relative to event onset. Together with tmax, 
                 defines the time window you extract as a data sample. For an event at time T, 
                 you grab MEG data from T + tmin up to T + tmax.
            tmax: End time of the sample in seconds relative to event onset. The number of timepoints 
                 per sample = int((tmax - tmin) * sfreq) where sfreq=250Hz. E.g., tmin=0, tmax=0.8 
                 yields 200 timepoints per sample.
            include_run_keys: List of specific sessions to include. Format per session: 
                            ('0', '1', 'Sherlock1', '1') = Subject 0, Session 1, Task Sherlock1, Run 1.
                            See pnpl.datasets.libribrain2025.constants.RUN_KEYS for all valid keys.
            exclude_run_keys: List of sessions to exclude (same format as include_run_keys).
            exclude_tasks: List of task names to exclude (e.g., ['Sherlock1']).
            standardize: Whether to z-score normalize each channel's MEG data using mean and std 
                        computed across all included runs. Formula: 
                        normalized_data[channel] = (raw_data[channel] - channel_mean[channel]) / channel_std[channel].
            clipping_boundary: If specified, clips all values to [-clipping_boundary, clipping_boundary]. 
                             Set to None for no clipping.
            channel_means: Pre-computed channel means for standardization. If None and standardize=True, 
                          will be computed from the dataset.
            channel_stds: Pre-computed channel standard deviations for standardization. If None and 
                         standardize=True, will be computed from the dataset.
            include_info: Whether to include additional info dict in each sample containing dataset name, 
                         subject, session, task, run, and onset time.
            preload_files: Whether to "eagerly" download all dataset files from HuggingFace when the 
                          dataset object is created (True) or "lazily" download files on demand (False). 
                          We recommend leaving this as True unless you have a specific reason not to.
            preload_event_file: Whether to preload event files specifically (only used if preload_files=True).
            download: Whether to download files from HuggingFace if not found locally (True) or 
                     throw an error if files are missing locally (False).

        Returns:
            Data samples with shape (channels, time) where channels=306 MEG channels and time 
            depends on (tmax-tmin)*sfreq.
        """
        os.makedirs(data_path, exist_ok=True)
        self.data_path = data_path
        self.partition = partition
        self.preprocessing_str = preprocessing_str
        self.tmin = tmin
        self.tmax = tmax
        self.include_run_keys = include_run_keys
        self.exclude_run_keys = exclude_run_keys
        self.standardize = standardize
        self.clipping_boundary = clipping_boundary
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.include_info = include_info
        self.preload_files = preload_files
        self.preload_event_file = preload_event_file
        self.download = download

        if partition is not None:
            if include_run_keys or exclude_run_keys or exclude_tasks:
                raise ValueError(
                    "partition is a shortcut to indicate what data to include. include_run_keys, exclude_run_keys, exclude_tasks must be empty when partition is not None")
            if partition == "train":
                exclude_run_keys = VALIDATION_RUN_KEYS + TEST_RUN_KEYS
            elif partition == "validation":
                include_run_keys = VALIDATION_RUN_KEYS
            elif partition == "test":
                include_run_keys = TEST_RUN_KEYS
            else:
                raise ValueError(
                    f"Invalid partition: {partition}. Must be one of: train, validation, test")
        # Convert channel_means and channel_stds to np.ndarray if they are provided as lists.
        if isinstance(channel_means, list):
            self.channel_means = np.array(channel_means)
        if isinstance(channel_stds, list):
            self.channel_stds = np.array(channel_stds)

        include_run_keys = [tuple(run_key) for run_key in include_run_keys]
        exclude_run_keys = [tuple(run_key) for run_key in exclude_run_keys]
        check_include_and_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS)

        intended_run_keys = include_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS)
        self.intended_run_keys = [
            run_key for run_key in intended_run_keys if run_key[2] not in exclude_tasks]

        if len(self.intended_run_keys) == 0:
            raise ValueError(
                f"Your configuration does not allow any run keys to be included. Please check configuration: include_run_keys={include_run_keys}, exclude_run_keys={exclude_run_keys}, exclude_tasks={exclude_tasks}"
            )

        # Preload files if requested BEFORE calling _get_sfreq which would trigger sequential downloads
        if self.preload_files and self.download:
            self.prefetch_files(get_event_files = self.preload_event_file)

        # Now we can safely get sfreq as files are already downloading/downloaded
        self.sfreq = self._get_sfreq(
            self.intended_run_keys[0][0],
            self.intended_run_keys[0][1],
            self.intended_run_keys[0][2],
            self.intended_run_keys[0][3]
        )
        self.points_per_sample = int((tmax - tmin) * self.sfreq)
        self.open_h5_datasets = {}

    def __len__(self):
        return len(self.samples)

    def prefetch_files(self, get_event_files = True):
        """Preload all required files in parallel."""
        futures = []
        needed_files = set()

        # Collect all file paths that we'll need
        for subject, session, task, run in self.intended_run_keys:
            # H5 files
            h5_path = self._get_h5_path(subject, session, task, run)
            if not os.path.exists(h5_path):
                needed_files.add(h5_path)

            if get_event_files:
                # Event files
                events_path = self._get_events_path(subject, session, task, run)
                if not os.path.exists(events_path):
                    needed_files.add(events_path)

        # Schedule downloads for all files that don't exist locally
        for fpath in needed_files:
            futures.append(self._schedule_download(fpath))

        # Wait for all downloads to complete
        if futures:
            print(f"Downloading {len(futures)} files...")
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading a file: {e}")
            print("Done!")

    def _schedule_download(self, fpath):
        """Schedule a file download with retry logic."""
        rel_path = os.path.relpath(fpath, self.data_path)
        # Windows fix: convert Windows path separator to URL path separator
        rel_path = rel_path.replace(os.path.sep, '/')
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with LibriBrainBase._lock:
            if fpath not in LibriBrainBase._download_futures:
                LibriBrainBase._download_futures[fpath] = LibriBrainBase._executor.submit(
                    LibriBrainBase._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    data_path=self.data_path
                )
            return LibriBrainBase._download_futures[fpath]

    def _ensure_file(self, fpath: str) -> str:
        """
        Ensures the file exists locally, downloading if needed.
        This is a blocking call that waits for download to complete.
        """
        if os.path.exists(fpath):
            return fpath

        if not self.download:
            raise FileNotFoundError(f"File not found: {fpath}. Download is disabled.")

        future = self._schedule_download(fpath)
        # Wait for the download to complete
        return future.result()

    @classmethod
    def ensure_file_download(cls, fpath: str, data_path: str) -> str:
        """
        Class method to download a file using the sophisticated LibriBrain download system
        without requiring dataset instantiation.
        
        Args:
            fpath: Full path to the file that should exist locally
            data_path: Base data directory for computing relative paths
            
        Returns:
            Path to the downloaded file
            
        This provides the same sophisticated download functionality as the full dataset:
        - Retry logic with exponential backoff
        - Multiple HuggingFace repository fallbacks  
        - Environment variable support (HF_TOKEN, HF_DATASET)
        - Thread-based downloading with deduplication
        """
        if os.path.exists(fpath):
            return fpath
            
        # Use the class-level download infrastructure
        rel_path = os.path.relpath(fpath, data_path)
        rel_path = rel_path.replace(os.path.sep, '/')
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with cls._lock:
            if fpath not in cls._download_futures:
                cls._download_futures[fpath] = cls._executor.submit(
                    cls._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    data_path=data_path
                )
            future = cls._download_futures[fpath]
        
        # Wait for download to complete
        return future.result()

    @staticmethod  
    def _download_with_retry_static(fpath, rel_path, data_path, max_retries=5):
        """
        Download a file with retry logic and multiple repository fallback.
        
        Tries repositories in this order:
        1. Custom dataset from HF_DATASET env var (if HF_TOKEN is provided) - PRIORITY
        2. pnpl/LibriBrain (public)
        3. pnpl/LibriBrain-Competition-2025 (public)
        
        Args:
            fpath: Full local path where file should be saved
            rel_path: Relative path within the HuggingFace repository
            data_path: Base data directory for the dataset
            max_retries: Maximum number of retry attempts per repository
            
        Returns:
            Path to the downloaded file
        """
        # Get optional environment variables
        hf_token = os.getenv('HF_TOKEN')
        hf_dataset = os.getenv('HF_DATASET')
        
        # Define repositories to try in order
        repos_to_try = []
        
        # Try custom dataset FIRST if both token and dataset are provided
        if hf_token and hf_dataset:
            print(f"Using custom dataset: {hf_dataset} (with token)")
            repos_to_try.append({"repo_id": hf_dataset, "token": hf_token})
        elif hf_dataset:
            print(f"⚠️ HF_DATASET set ({hf_dataset}) but HF_TOKEN not found - will try public repos only")
        elif hf_token:
            print(f"⚠️ HF_TOKEN set but HF_DATASET not specified - will try public repos only")
        
        # Then try public repositories as fallback
        repos_to_try.extend([
            {"repo_id": "pnpl/LibriBrain", "token": None},
            {"repo_id": "pnpl/LibriBrain-Competition-2025", "token": None}
        ])
        
        last_exception = None
        
        for repo_config in repos_to_try:
            retries = 0
            while retries < max_retries:
                try:
                    download_kwargs = {
                        "repo_id": repo_config["repo_id"],
                        "repo_type": "dataset",
                        "filename": rel_path,
                        "local_dir": data_path
                    }
                    
                    # Add token if provided
                    if repo_config["token"]:
                        download_kwargs["token"] = repo_config["token"]
                    
                    return hf_hub_download(**download_kwargs)
                    
                except (RepositoryNotFoundError, EntryNotFoundError, GatedRepoError) as e:
                    # These are permanent errors - don't retry, move to next repo immediately
                    last_exception = e
                    break  # Move to next repository immediately
                    
                except (ConnectionError, Timeout, HTTPError) as e:
                    # These are potentially temporary errors - retry with backoff
                    last_exception = e
                    retries += 1
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    if retries < max_retries:
                        print(f"Network error for {os.path.basename(fpath)} from {repo_config['repo_id']}, retrying in {wait_time:.1f}s ({retries}/{max_retries}): {type(e).__name__}")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to download {os.path.basename(fpath)} from {repo_config['repo_id']} after {max_retries} network retries")
                        break  # Move to next repository
                        
                except Exception as e:
                    # Unknown errors - be conservative and retry a few times
                    last_exception = e
                    retries += 1
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    if retries < max_retries:
                        print(f"Unknown error for {os.path.basename(fpath)} from {repo_config['repo_id']}, retrying in {wait_time:.1f}s ({retries}/{max_retries}): {type(e).__name__}")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to download {os.path.basename(fpath)} from {repo_config['repo_id']} after {max_retries} attempts: {type(e).__name__}")
                        break  # Move to next repository
        
        # If we've exhausted all repositories, raise the last exception
        print(f"File {os.path.basename(fpath)} not found in any of the {len(repos_to_try)} repositories")
        raise last_exception

    def _get_h5_path(self, subject: str, session: str, task: str, run: str) -> str:
        """
        Gets the path to the h5 file.
        """
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
        if self.preprocessing_str is not None:
            fname += f"_proc-{self.preprocessing_str}"
        fname += "_meg.h5"
        return os.path.join(self.data_path, task, "derivatives", "serialised", fname)

    def _get_events_path(self, subject: str, session: str, task: str, run: str) -> str:
        """
        Gets the path to the events file.
        """
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        return os.path.join(self.data_path, task, "derivatives", "events", fname)

    def _ids_to_h5_path(self, subject: str, session: str, task: str, run: str) -> str:
        """
        Gets the path to the h5 file and ensures it exists.
        """
        path = self._get_h5_path(subject, session, task, run)
        return self._ensure_file(path)

    def _get_sfreq(self, subject, session, task, run):
        h5_path = self._ids_to_h5_path(subject, session, task, run)
        with h5py.File(h5_path, "r") as h5_file:
            sfreq = h5_file.attrs["sample_frequency"]
        return sfreq

    def _load_events(self, subject: str, session: str, task: str, run: str):
        fpath = self._get_events_path(subject, session, task, run)
        fpath = self._ensure_file(fpath)
        events_df = pd.read_csv(fpath, sep="\t")
        return events_df

    def _calculate_standardization_params(self):
        n_samples = []
        means = []
        stds = []
        for run_key in self.run_keys:
            subject, session, task, run = run_key
            hdf_dataset = h5py.File(self._ids_to_h5_path(
                subject, session, task, run), "r")["data"]

            if "channel_means" in hdf_dataset.attrs and "channel_stds" in hdf_dataset.attrs:
                channel_means = hdf_dataset.attrs["channel_means"]
                channel_stds = hdf_dataset.attrs["channel_stds"]
            else:
                data = hdf_dataset[:, :]
                channel_means = np.mean(data, axis=1)
                channel_stds = np.std(data, axis=1)
                hdf_dataset.file.close()
                with h5py.File(self._ids_to_h5_path(subject, session, task, run), "r+") as f:
                    f["data"].attrs["channel_means"] = channel_means
                    f["data"].attrs["channel_stds"] = channel_stds
                hdf_dataset = h5py.File(self._ids_to_h5_path(
                    subject, session, task, run), "r")["data"]
                print("calculated stats for: ", run_key)

            n_samples.append(hdf_dataset.shape[1])
            means.append(channel_means)
            stds.append(channel_stds)
        means = np.array(means)
        stds = np.array(stds)
        n_samples = np.array(n_samples)
        self.channel_stds, self.channel_means = self._accumulate_stds(
            means, stds, n_samples)
        self.broadcasted_stds = np.tile(
            self.channel_stds, (self.points_per_sample, 1)).T
        self.broadcasted_means = np.tile(
            self.channel_means, (self.points_per_sample, 1)).T

    @staticmethod
    def _accumulate_stds(ch_means, ch_stds, n_samples):
        """
        ch_means: np.ndarray (n_groups, n_channels)
        ch_stds: np.ndarray (n_groups, n_channels)
        n_samples: np.ndarray (n_groups)
        """
        vars = np.array(ch_stds) ** 2
        means_total = np.average(ch_means, axis=0, weights=n_samples)
        sum_of_squares_within = np.sum(
            vars * np.tile(n_samples, (vars.shape[1], 1)).T, axis=0)
        sum_of_squares_between = np.sum(
            (ch_means - np.tile(means_total, (ch_means.shape[0], 1))) ** 2 *
            np.tile(n_samples, (ch_means.shape[1], 1)).T,
            axis=0
        )
        sum_of_squares_total = sum_of_squares_within + sum_of_squares_between
        return np.sqrt(sum_of_squares_total / np.sum(n_samples)), means_total

    def _clip_sample(self, sample, boundary):
        sample = np.clip(sample, -boundary, boundary)
        return sample

    def __getitem__(self, idx):
        # returns channels x time
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.samples)}"
            )
        sample = self.samples[idx]
        subject, session, task, run, onset, label = sample
        if self.include_info:
            info = {
                "dataset": "libribrain2025",
                "subject": subject,
                "session": session,
                "task": task,
                "run": run,
                "onset": torch.tensor(onset, dtype=torch.float32),
            }

        if (subject, session, task, run) not in self.open_h5_datasets:
            h5_path = self._ids_to_h5_path(subject, session, task, run)
            h5_dataset = h5py.File(h5_path, "r")["data"]
            self.open_h5_datasets[(subject, session, task, run)] = h5_dataset
        else:
            h5_dataset = self.open_h5_datasets[(subject, session, task, run)]

        start = max(0, int((onset + self.tmin) * self.sfreq))
        end = start + self.points_per_sample
        data = h5_dataset[:, start:end]

        if self.standardize:
            # for the edge case in which the last samples are smaller than points_per_sample,
            if data.shape[1] < self.broadcasted_means.shape[1]:
                self.broadcasted_means = self.broadcasted_means[:,0:data.shape[1]]
                self.broadcasted_stds = self.broadcasted_stds[:,0:data.shape[1]]

            data = (data - self.broadcasted_means) / self.broadcasted_stds

        if self.clipping_boundary is not None:
            data = self._clip_sample(data, self.clipping_boundary)

        if self.include_info:
            return [torch.tensor(data, dtype=torch.float32), label, info]
        return [torch.tensor(data, dtype=torch.float32), label, {}]
