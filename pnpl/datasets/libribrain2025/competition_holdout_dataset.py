import os
from torch.utils.data import Dataset
import torch
import csv
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from pnpl.datasets import LibriBrainSpeech
from pnpl.datasets.libribrain2025.constants import PHONEME_CLASSES, SPEECH_OUTPUT_DIM, PHONEME_HOLDOUT_PREDICTIONS, SPEECH_HOLDOUT_PREDICTIONS
from pnpl.datasets.libribrain2025.speech_dataset_holdout import LibriBrainSpeechHoldout
from pnpl.datasets.libribrain2025.base import LibriBrainBase

class LibriBrainCompetitionHoldout(Dataset):
    def __init__(self, data_path,
                 tmin: float = 0.0,
                 tmax: float = 0.8,
                 standardize=True,
                 clipping_boundary=10,
                 stride=1,
                 task: str = "speech",
                 channel_means: np.ndarray | None = None,
                 channel_stds: np.ndarray | None = None,
                 download: bool = True):
        """
        LibriBrain 2025 Competition Holdout Dataset.

        This dataset provides access to the holdout data used for the LibriBrain 2025 competition.
        It loads the specific holdout session and prepares it for generating competition submissions.

        Args:
            data_path: Path where you wish to store the dataset. The local dataset structure 
                      will follow the same BIDS-like structure as the HuggingFace repo:
                      ```
                      data_path/
                      ├── {task}/                    # e.g., "Sherlock1", "COMPETITION_HOLDOUT"  
                      │   └── derivatives/
                      │       ├── serialised/       # MEG data files
                      │       │   └── sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5
                      │       └── events/            # Event files  
                      │           └── sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv
                      ```
            tmin: Start time of the sample in seconds relative to the onset/window start. 
                 For speech task: Together with tmax, defines the time window size for each sample.
                 For phoneme task: Must be >= 0.0. Phoneme samples are 0.5s long; smaller windows are supported.
            tmax: End time of the sample in seconds relative to the onset/window start. 
                 For speech task: The number of timepoints per sample = int((tmax - tmin) * sfreq) where sfreq=250Hz.
                 For phoneme task: Must be <= 0.5. Phoneme samples are 0.5s long; smaller windows are supported.
            standardize: Whether to z-score normalize each channel's MEG data using mean and std 
                        computed from the holdout data samples.
            clipping_boundary: If specified, clips all values to [-clipping_boundary, clipping_boundary]. 
                             This can help with outliers. Set to None for no clipping.
            stride: Controls how far (in time) you move the sliding window between consecutive samples. 
                   Only used for speech task. Phoneme task has pre-defined, event-aligned samples.
            task: Type of task for the competition. Currently, "speech" and "phoneme" are supported.
                 "speech": speech vs silence classification
                 "phoneme": phoneme classification (39 classes)
            channel_means: Pre-computed channel means for standardization. If None and standardize=True,
                          will compute from the loaded holdout data.
            channel_stds: Pre-computed channel standard deviations for standardization.
            download: Whether to download files from HuggingFace if not found locally (True) or 
                     throw an error if files are missing locally (False).

        Note:
            ⚠️ This dataset loads the specific holdout session ('0', '2025', 'COMPETITION_HOLDOUT', '1') 
            for speech and ('0', '2025', 'COMPETITION_HOLDOUT', '2') for phoneme that are used for 
            competition evaluation.
            
            When making predictions, ensure your final submission matches the expected number of 
            timepoints. Use the generate_submission_in_csv() method to create properly formatted 
            submission files.

        Returns:
            Data samples with shape (channels, time) where channels=306 MEG channels.
        """
        # Path to the data
        self.data_path = data_path
        self.task = task
        self.dataset = None
        
        # Adjust default tmax for phoneme task if needed
        if task == "phoneme" and tmax == 0.8:
            print("Note: Adjusting tmax for phoneme task to 0.5s")
            tmax = 0.5  # Phoneme samples are only 0.5s long
            
        self.tmin = tmin
        self.tmax = tmax
        self.standardize = standardize
        self.clipping_boundary = clipping_boundary
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.sfreq = 250  # MEG sampling frequency
        
        if task == "speech":
            try:
                self.dataset = LibriBrainSpeechHoldout(
                    data_path=self.data_path,
                    tmin = tmin,
                    tmax = tmax,
                    include_run_keys=[("0", "2025", "COMPETITION_HOLDOUT", "1")],
                    standardize=standardize,
                    clipping_boundary=clipping_boundary,
                    preprocessing_str="bads+headpos+sss+notch+bp+ds",
                    preload_files=False,
                    include_info=True,
                    stride=stride,
                    download=download,
                    channel_means=channel_means,
                    channel_stds=channel_stds
                )
                self.samples = self.dataset.samples
            except Exception as e:
                warnings.warn(f"Failed to load speech dataset: {e}")
                raise RuntimeError("Failed to load speech dataset. Check the data path and parameters.")
        elif task == "phoneme":
            # Validate tmin/tmax for phoneme task
            if tmin < 0.0:
                raise ValueError(f"tmin must be >= 0.0 for phoneme task, got {tmin}")
            if tmax > 0.5:
                raise ValueError(f"tmax must be <= 0.5 for phoneme task (samples are 0.5s long), got {tmax}")
            if tmin >= tmax:
                raise ValueError(f"tmin must be < tmax, got tmin={tmin}, tmax={tmax}")
            
            # Calculate time window parameters
            self.points_per_sample = int((tmax - tmin) * self.sfreq)
            
            # Define the expected path for the phoneme holdout data (consistent with base class pattern)
            phoneme_filename = "sub-0_ses-2025_task-COMPETITION_HOLDOUT_run-2_proc-bads+headpos+sss+notch+bp+ds_meg.pt"
            phoneme_file_path = os.path.join(data_path, "COMPETITION_HOLDOUT", "derivatives", "serialised", phoneme_filename)
            
            # Use base class download functionality
            if download:
                phoneme_file_path = LibriBrainBase.ensure_file_download(phoneme_file_path, data_path)
            elif not os.path.exists(phoneme_file_path):
                raise FileNotFoundError(
                    f"Phoneme holdout data not found at {phoneme_file_path}. "
                    f"Please ensure the file exists or set download=True to download from HuggingFace."
                )
            
            # Load the phoneme samples
            try:
                self.raw_samples = torch.load(phoneme_file_path, map_location='cpu', weights_only=True)
                
                # Validate data structure
                if not isinstance(self.raw_samples, torch.Tensor):
                    if hasattr(self.raw_samples, '__len__') and len(self.raw_samples) > 0:
                        self.raw_samples = torch.stack([s for s in self.raw_samples])
                    else:
                        raise RuntimeError(f"Invalid data structure in {phoneme_file_path}")
                
                # Phoneme holdout: saved tensors are already standardized during generation.
                # Disable standardization here to avoid double-standardization.
                if self.standardize or (channel_means is not None or channel_stds is not None):
                    warnings.warn(
                        "Standardization parameters are ignored for phoneme holdout; "
                        "data is assumed pre-standardized. Setting standardize=False.")
                self.standardize = False
                    
            except Exception as e:
                warnings.warn(f"Failed to load phoneme dataset: {e}")
                raise RuntimeError(f"Failed to load phoneme dataset from {phoneme_file_path}. Error: {e}")
        else:
            raise NotImplementedError(f"Task '{task}' is not supported. Please use 'speech' or 'phoneme'.")

    def _calculate_standardization_params(self):
        """Calculate standardization parameters from phoneme holdout data, following base class pattern."""
        print("Computing standardization parameters from phoneme holdout data...")
        # Compute channel-wise statistics across all samples
        # Shape: (n_samples, n_channels, n_timepoints) -> (n_channels,)
        flattened = self.raw_samples.view(self.raw_samples.shape[0], self.raw_samples.shape[1], -1)
        flattened = flattened.permute(1, 0, 2).contiguous().view(self.raw_samples.shape[1], -1)
        
        self.channel_means = flattened.mean(dim=1).numpy()
        self.channel_stds = flattened.std(dim=1).numpy()
        
        # Setup broadcasted arrays for efficient processing (following base class pattern)
        self.broadcasted_stds = np.tile(
            self.channel_stds, (self.points_per_sample, 1)).T
        self.broadcasted_means = np.tile(
            self.channel_means, (self.points_per_sample, 1)).T

    def _clip_sample(self, sample, boundary):
        """Clip sample values to [-boundary, boundary], following base class pattern."""
        sample = np.clip(sample, -boundary, boundary)
        return sample

    def generate_submission_in_csv(self, predictions, output_path: str):
        """
        Generates a submission file in CSV format for the LibriBrain competition.
        The file contains the run keys and the corresponding labels.
        Args:
            predictions (List[Tensor]): 
                - For speech: List of scalar tensors, each representing a speech probability.
                - For phoneme: List of 39-dimensional tensors, each representing phoneme probabilities.
            output_path (str): Path to save the CSV file.
        """
        if self.task == "speech":
            if len(predictions) != SPEECH_HOLDOUT_PREDICTIONS:
                raise (ValueError(
                    f"Length of speech predictions ({len(predictions)}) does not match the expected number of segments ({SPEECH_HOLDOUT_PREDICTIONS})."))
            if predictions[0].shape[0] != SPEECH_OUTPUT_DIM:
                raise (ValueError(
                    f"Speech prediction dimension {predictions[0].shape[0]} does not match expected size (1 for scalar probability)."))
            with open(output_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["idx", "speech_prob"])

                for idx, tensor in enumerate(predictions):
                    # Ensure we extract the scalar float from tensor
                    speech_prob = tensor.item() if isinstance(
                        tensor, torch.Tensor) else float(tensor)
                    writer.writerow([idx, speech_prob])
        elif self.task == "phoneme":
            if len(predictions) != PHONEME_HOLDOUT_PREDICTIONS:
                raise (ValueError(
                    f"Length of phoneme predictions ({len(predictions)}) does not match the expected number of segments ({PHONEME_HOLDOUT_PREDICTIONS})."))
            if predictions[0].shape[0] != PHONEME_CLASSES:
                raise (ValueError(
                    f"Phoneme classes dimension {predictions[0].shape[0]} does not match expected size ({PHONEME_CLASSES})."))
            
            with open(output_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Create header: segment_idx, phoneme_1, ..., phoneme_39
                header = ["segment_idx"] + \
                    [f"phoneme_{i + 1}" for i in range(PHONEME_CLASSES)]
                writer.writerow(header)

                for idx, tensor in enumerate(predictions):
                    # Ensure tensor is a flat list of floats
                    if isinstance(tensor, torch.Tensor):
                        probs = tensor.squeeze().tolist()  # shape: (39,)
                    else:
                        # if tensor is already a list-like
                        probs = list(tensor)

                    writer.writerow([idx] + probs)


    def speech_labels(self):
        return self.dataset.speech_labels if self.task == "speech" else None

    def __len__(self):
        if self.task == "speech":
            return len(self.dataset.samples)
        else:  # phoneme
            return len(self.raw_samples)

    def __getitem__(self, idx):
        # returns channels x time
        if self.task == "speech":
            return self.dataset[idx]
        else:  # phoneme
            # Extract the time window from the full sample (following base class pattern)
            # Each sample is 125 timepoints (0.5s), we extract [tmin:tmax] window
            start_idx = int(self.tmin * self.sfreq)
            end_idx = start_idx + self.points_per_sample
            
            # Get the sample and slice the time dimension
            data = self.raw_samples[idx][:, start_idx:end_idx].numpy()
            
            # Apply standardization if requested (following base class pattern)
            if self.standardize:
                # Handle edge case where data might be smaller than expected
                if data.shape[1] < self.broadcasted_means.shape[1]:
                    broadcasted_means = self.broadcasted_means[:, 0:data.shape[1]]
                    broadcasted_stds = self.broadcasted_stds[:, 0:data.shape[1]]
                else:
                    broadcasted_means = self.broadcasted_means
                    broadcasted_stds = self.broadcasted_stds
                
                data = (data - broadcasted_means) / broadcasted_stds
            
            # Apply clipping if requested (following base class pattern)
            if self.clipping_boundary is not None:
                data = self._clip_sample(data, self.clipping_boundary)
            
            return torch.tensor(data, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage for speech task
    print("Testing speech task...")
    speech_dataset = LibriBrainCompetitionHoldout(
        data_path = "/tmp/libribrain_test",
        tmax=0.8,
        task="speech",
        download=False)  # Set to True if you want to download

    print(f"Speech dataset: {len(speech_dataset)} samples")
    
    # Example usage for phoneme task
    print("\nTesting phoneme task...")
    phoneme_dataset = LibriBrainCompetitionHoldout(
        data_path = "/tmp/libribrain_test",
        task="phoneme",
        download=False)  # Set to True if you want to download
    
    print(f"Phoneme dataset: {len(phoneme_dataset)} samples")
    if len(phoneme_dataset) > 0:
        print(f"Sample shape: {phoneme_dataset[0].shape}")
        
    # Example of generating a submission
    if len(phoneme_dataset) > 0:
        print("\nGenerating example phoneme submission...")
        random_predictions = [torch.softmax(torch.randn(PHONEME_CLASSES), dim=0) for _ in range(10)]  # Just first 10 for demo
        # In practice, you'd generate predictions for all samples:
        # random_predictions = [torch.softmax(torch.randn(PHONEME_CLASSES), dim=0) for _ in range(len(phoneme_dataset))]
        
        # Note: This will fail unless you have all predictions, just showing the format
        # phoneme_dataset.generate_submission_in_csv(random_predictions, "phoneme_submission_demo.csv")