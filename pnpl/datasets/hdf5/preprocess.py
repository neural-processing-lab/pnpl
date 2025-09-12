import mne
import numpy as np
import json
import mne_bids
import os
import warnings
import h5py


def preprocess_raw(raw: mne.io.Raw, config: dict):
    """
    Preprocess a single MNE Raw object based on the provided configuration.

    Parameters:
    - config: dict, config specifying preprocessing steps.
    - raw: mne.io.Raw, raw data to preprocess.

    Returns:
    - raw: mne.io.Raw, preprocessed raw data.
    """
    print("Starting preprocessing")

    if (config is None or "preproc" not in config):
        warnings.warn("No preprocessing steps for raw data provided")
        return raw
    # Apply each preprocessing step as defined in the config
    for step in config["preproc"]:
        for method, params in step.items():
            if method == 'pick':
                raw = raw.pick(**params)
            elif method == 'filter':
                raw.filter(**params)
            elif method == 'notch_filter':
                if ("freqs" not in params):
                    raise ValueError(
                        "Frequency values not provided for notch filter")
                raw.notch_filter(**params)
            elif method == 'resample':
                if ("sfreq" not in params):
                    raise ValueError("Resample frequency not provided")
                raw.resample(**params)
            elif method == 'find_bad_channels_maxwell':
                bads = mne.preprocessing.find_bad_channels_maxwell(
                    raw, **params)
                raw.info['bads'] = bads
            elif method == 'interpolate_bads':
                raw.interpolate_bads(**params)

            elif method == 'pick_types':
                raise ValueError(
                    "Pick types is not supported please use pick instead")
            elif method == "bad_channels":
                raise ValueError(
                    "Bad channels is not supported please use find_bad_channels_maxwell instead")
            else:
                raise ValueError(f"Unknown method {method}")

    print("Instance preprocessed")
    return raw


def get_raw(data_path: str, subject: str, session: str, task: str, run: str, processing: str):
    raw_bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        processing=processing,
        root=data_path,
    )
    try:
        raw = mne.io.read_raw(raw_bids_path.fpath)
        raw.load_data()
    except FileNotFoundError:
        print("missing", "subject: ", subject, "session: ",
              session, "task: ", task, "run: ", run)
        return None
    return raw


def get_sensor_positions(raw):
    sensor_positions = []
    for ch in raw.info["chs"]:
        # Extracts the first three elements: X, Y, Z
        pos = ch["loc"][:3]
        sensor_positions.append(pos.tolist())
    return sensor_positions


def get_info(raw: mne.io.Raw, subject: str, session: str, task: str, run: str, subject_idx: int):
    sensor_positions = get_sensor_positions(raw)
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    channel_means = np.mean(data, axis=1)
    channel_stds = np.std(data, axis=1)
    info = {
        "subject": subject,
        "session": session,
        "subject_idx": subject_idx,
        "task": task,
        "run": run,
        "dataset": "Sherlock1",
        "sfreq": sfreq,
        "sensor_xyz": sensor_positions,
        "channel_means": channel_means.tolist(),
        "channel_stds": channel_stds.tolist(),
        "n_samples": data.shape[1]
    }
    return info


def write_h5(data: np.ndarray, output_path: str, subject: str, session: str, task: str, run: str, processing: str, info: dict):
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        run=run,
        datatype="meg",
        processing=processing,
        root=output_path,
    )
    os.makedirs(bids_path.directory, exist_ok=True)
    file_path = os.path.join(bids_path.directory, bids_path.basename + ".h5")
    with h5py.File(file_path, "w") as f:
        ds = f.create_dataset("data", data=data, dtype=np.float32,
                              chunks=((data.shape[0], 40)))
        for key, value in info.items():
            if (value is None):
                continue
            ds.attrs[key] = value
    print(
        f"Serialized: sub-{subject} ses-{session} task-{task} run-{run} to {file_path}"
    )


def serialize_dataset_to_h5(
    data_path: str,
    subjects: list[str],
    sessions: list[str],
    tasks: list[str],
    runs: list[str | None] = [None],
    output_path: str | None = None,
    preprocess_config: dict = {},
    preprocessing_name: str | None = None,
    subjects_for_indexing: list[str] | None = None
):

    if (output_path is None):
        output_path = os.path.join(
            data_path, "derivatives", "serialized", "default")
    # dump config
    os.makedirs(output_path, exist_ok=True)
    preproc_args = {
        "data_path": data_path,
        "preprocess_config": preprocess_config,
        "include_subjects": subjects,
        "include_sessions": sessions,
    }
    json.dump(preproc_args, open(os.path.join(
        output_path, "preproc_args.json"), "w"))

    for subject in subjects:
        for session in sessions:
            for task in tasks:
                for run in runs:
                    raw = get_raw(data_path, subject, session,
                                  task, run, preprocessing_name)
                    if raw is None:
                        print(
                            f"Missing subject: {subject} session: {session} task: {task} run: {run}")
                        continue
                    raw = preprocess_raw(raw, preprocess_config)
                    subject_idx = subjects_for_indexing.index(subject)
                    info = get_info(raw, subject, session,
                                    task, run, subject_idx)
                    data = raw.get_data()
                    write_h5(data, output_path,
                             subject, session, task, run=run, processing=preprocessing_name, info=info)
    print(f"Serialization complete for subjects"
          + f"{subjects} and sessions {sessions} and tasks {tasks} and runs {runs}")
