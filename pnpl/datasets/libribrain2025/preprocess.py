# conda activate mne

import os
import mne
import h5py
import numpy as np
from time import time
import sys


def ftime(f):
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        print(f"{f.__name__} executed in {time() - start:.4f} seconds")
        return result
    return wrapper


@ftime
# def fif2h5(fif_file, dtype=None, output_dir=None):
def fif2h5(fif_file, dtype=None, output_dir=None, chunk_size=50, compression="gzip", compression_opts=4):
    raw = mne.io.read_raw_fif(fif_file, preload=True)

    if len(raw.info['bads']) > 0:
        print(f"Error: {fif_file} contains bad channels: {raw.info['bads']}")
        sys.exit(1)

    times = raw.times
    meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
    data = raw.get_data(picks=meg_picks)

    if dtype is None:
        print(
            f"Default saving times as {times.dtype} and data as {data.dtype}")
    else:
        print(f"Converting times and data to {dtype}")  # e.g. np.float32
        times = times.astype(dtype)
        data = data.astype(dtype)

    channel_names = [raw.ch_names[idx] for idx in meg_picks]
    channel_types = [mne.io.pick.channel_type(
        raw.info, idx) for idx in meg_picks]

    h5_file = os.path.splitext(fif_file)[0] + ".h5"
    if output_dir is not None:
        print(f'Saving h5 files to {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        h5_file = os.path.join(output_dir, h5_file)

    with h5py.File(h5_file, "w") as f:
        # f.create_dataset("data", data=data)
        # f.create_dataset("times", data=times)
        #        chunk_size=50, compression="gzip", compression_opts=4
        if compression:
            f.create_dataset("data", data=data, compression=compression,
                             compression_opts=compression_opts, chunks=(data.shape[0], chunk_size))
            f.create_dataset("times", data=times, compression=compression,
                             compression_opts=compression_opts, chunks=(chunk_size,))
        else:
            f.create_dataset("data", data=data, chunks=(
                data.shape[0], chunk_size))
            f.create_dataset("times", data=times, chunks=(chunk_size,))

#        f.create_dataset("data", data=data, compression=compression, compression_opts=compression_opts, chunks=(data.shape[0], chunk_size))
#        f.create_dataset("times", data=times, compression=compression, compression_opts=compression_opts, chunks=(chunk_size,))

        # metadata
        f.attrs["sample_frequency"] = raw.info["sfreq"]
        f.attrs["highpass_cutoff"] = raw.info['highpass']
        f.attrs["lowpass_cutoff"] = raw.info['lowpass']
        f.attrs["channel_names"] = ", ".join(channel_names)
#        f.attrs["channel_types"] = np.string_(channel_types)
        f.attrs["channel_types"] = ", ".join(channel_types)

    return h5_file


start = time()

fif_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".fif")]
for i, fif_file in enumerate(fif_files):
    if os.path.exists(os.path.splitext(fif_file)[0] + ".h5"):
        print(f"skipping {fif_file}")
        continue
    print(
        f"starting {fif_file} {i+1} of {len(fif_files)}: getting data, times, metadata; saving to h5")
#    fif2h5(fif_file)
#    fif2h5(fif_file, dtype=np.float32, output_dir='h5float32')
    fif2h5(fif_file, dtype=np.float32, output_dir='h5float32_chunk50',
           chunk_size=50, compression=None)
#    fif2h5(fif_file, dtype=np.float32, output_dir='h5float32_chunk50_gzip', chunk_size=50, compression="gzip")

# fif_file = "sub-0_ses-1_task-Sherlock4_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.fif"
# fif2h5(fif_file)

print(f"Total run time of script: {time() - start:4f} seconds")
