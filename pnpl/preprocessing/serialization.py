"""
Serialization utilities for converting MNE data to H5 format.

Provides functions to convert:
- Continuous Raw data to H5 (LibriBrain-style)
- Epoched data to H5 (MegNIST-style)
"""

import os
from typing import Optional, TYPE_CHECKING
import numpy as np
import h5py

if TYPE_CHECKING:
    import mne


def fif_to_h5(
    raw: "mne.io.Raw",
    output_path: str,
    dtype: np.dtype = np.float32,
    chunk_size: int = 50,
    compression: Optional[str] = None,
    compression_opts: int = 4,
) -> str:
    """
    Convert MNE Raw data to H5 format.
    
    Creates an H5 file with structure:
    - data: (channels, time) - MEG data
    - times: (time,) - Time vector
    - Attributes: sample_frequency, highpass_cutoff, lowpass_cutoff,
                  channel_names, channel_types
    
    Args:
        raw: MNE Raw object (preprocessed)
        output_path: Output H5 file path
        dtype: Data type for storage (default: float32)
        chunk_size: Chunk size for H5 datasets
        compression: Compression algorithm ('gzip' or None)
        compression_opts: Compression level (1-9)
        
    Returns:
        Path to created H5 file
    """
    import mne
    
    # Check for bad channels
    if len(raw.info['bads']) > 0:
        raise ValueError(f"Raw data contains bad channels: {raw.info['bads']}")
    
    # Extract data
    times = raw.times.astype(dtype)
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    data = raw.get_data(picks=meg_picks).astype(dtype)
    
    # Get channel info
    channel_names = [raw.ch_names[idx] for idx in meg_picks]
    # MNE's channel type helper moved between versions; prefer the stable APIs.
    try:
        channel_types = raw.get_channel_types(picks=meg_picks)
    except Exception:
        channel_types = [mne.channel_type(raw.info, idx) for idx in meg_picks]
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write H5 file
    with h5py.File(output_path, "w") as f:
        # Data datasets
        if compression:
            f.create_dataset(
                "data", data=data,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(data.shape[0], chunk_size),
            )
            f.create_dataset(
                "times", data=times,
                compression=compression,
                compression_opts=compression_opts,
                chunks=(chunk_size,),
            )
        else:
            f.create_dataset(
                "data", data=data,
                chunks=(data.shape[0], chunk_size),
            )
            f.create_dataset(
                "times", data=times,
                chunks=(chunk_size,),
            )
        
        # Metadata attributes
        f.attrs["sample_frequency"] = raw.info["sfreq"]
        f.attrs["highpass_cutoff"] = raw.info["highpass"]
        f.attrs["lowpass_cutoff"] = raw.info["lowpass"]
        f.attrs["channel_names"] = ", ".join(channel_names)
        f.attrs["channel_types"] = ", ".join(channel_types)
    
    return output_path


def epochs_to_h5(
    epochs: "mne.Epochs",
    output_path: str,
    dtype: np.dtype = np.float32,
    compression: Optional[str] = None,
    compression_opts: int = 4,
) -> str:
    """
    Convert MNE Epochs to H5 format.
    
    Creates an H5 file with structure:
    - data: (trials, channels, time) - Epoched MEG data
    - labels: (trials,) - Event labels
    - times: (time,) - Time vector
    - channel_names: (channels,) - Channel names
    - channel_types: (channels,) - Channel types
    - sensor_xyz: (channels, 3) - Sensor positions
    
    Args:
        epochs: MNE Epochs object
        output_path: Output H5 file path
        dtype: Data type for storage (default: float32)
        compression: Compression algorithm ('gzip' or None)
        compression_opts: Compression level (1-9)
        
    Returns:
        Path to created H5 file
    """
    import mne
    
    # Extract data
    data = epochs.get_data().astype(dtype)  # (trials, channels, time)
    times = epochs.times.astype(dtype)
    
    # Get labels from events (third column is event ID)
    labels = epochs.events[:, 2].astype(np.int32)
    
    # Get channel info
    meg_picks = mne.pick_types(epochs.info, meg=True, eeg=False, eog=False)
    channel_names = [epochs.ch_names[idx] for idx in meg_picks]
    try:
        channel_types = epochs.get_channel_types(picks=meg_picks)
    except Exception:
        channel_types = [mne.channel_type(epochs.info, idx) for idx in meg_picks]
    
    # Get sensor positions
    locs = np.array([epochs.info['chs'][idx]['loc'][:3] for idx in meg_picks])
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write H5 file
    with h5py.File(output_path, "w") as f:
        # Main datasets
        if compression:
            f.create_dataset("data", data=data, compression=compression, 
                           compression_opts=compression_opts)
            f.create_dataset("times", data=times, compression=compression,
                           compression_opts=compression_opts)
        else:
            f.create_dataset("data", data=data)
            f.create_dataset("times", data=times)
        
        f.create_dataset("labels", data=labels)
        
        # Channel info (stored as bytes for compatibility)
        f.create_dataset("channel_names", 
                        data=np.array(channel_names, dtype='S10'))
        f.create_dataset("channel_types",
                        data=np.array(channel_types, dtype='S15'))
        f.create_dataset("sensor_xyz", data=locs.astype(dtype))
    
    return output_path
