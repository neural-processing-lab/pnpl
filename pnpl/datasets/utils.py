from pathlib import Path
import os
import mne
import pandas as pd
import glob
from typing import Callable
import warnings


def get_bids_paths(data_path: str, path_to_id: Callable[[str], str | None], expected_ids: list[str], include_ids: list[str] | None = None, allow_non_id_paths: bool = True):
    """
    Given a folder of BIDS data, expected ids of subfolders, and a function that maps a subfolder path to an ID this method returns the paths and ids of the subfolders

    path_to_id: Function that converts a path to an id. Should return None if a path should not be considered. Operates on the full path.
    include_ids: List of ids to include. If None, include all.
    allow_non_id_paths: Decides how to handle paths that cannot be converted to an id. If False, an error is raised. Else they are ignored.

    Returns (paths: list, corresponding_ids: list)
    """

    instance_ids_on_disk = []
    instance_paths_on_disk = []
    for path in glob.glob(f"{data_path}/*"):
        instance_id = path_to_id(path)
        if (instance_id is None):
            if (not allow_non_id_paths):
                raise ValueError(
                    f"Encountered a path that could not be converted to an id: {path}")
            continue
        if (include_ids is None or instance_id in include_ids):
            instance_ids_on_disk.append(instance_id)
            instance_paths_on_disk.append(path)
        else:
            raise ValueError(
                f"Encountered an unexpected id: {instance_id}")

    if set(instance_ids_on_disk) != set(expected_ids):
        warnings.warn(
            f"Not all expected instances are in {data_path}. Missing instances: {set(expected_ids) - set(instance_ids_on_disk)}")

    return instance_paths_on_disk, instance_ids_on_disk


def get_cache_path(preproc_path, subject, session, task, run=None, proc=None):
    if session:
        fname = f"sub-{subject}_ses-{session}_task-{task}"
        cache_path = (
            str(preproc_path)
            + f"/preproc/sub-{subject}/sub-{subject}_ses-{session}_task-{task}"
        )
    else:
        fname = f"sub-{subject}_task-{task}"
        cache_path = (
            str(preproc_path) +
            f"/preproc/sub-{subject}/sub-{subject}_task-{task}"
        )
    if run:
        cache_path += f"_run-{run}"
        fname += f"_run-{run}"
    if proc:
        cache_path += f"_proc-{proc}"
        fname += f"_proc-{proc}"
    cache_path += "_meg"
    fname += '_meg_preproc_raw.fif'
    return cache_path + "/" + fname


def read_events_file(bids_root, subject, session, task, run=None):
    filename = f"sub-{subject}_ses-{session}_task-{task}"
    if run:
        filename += f"_run-{run}"
    filename += "_events.tsv"
    return pd.read_csv(
        Path(bids_root)
        / f"sub-{subject}/ses-{session}/meg/"
        / f"{filename}",
        sep="\t",
    )


def read_cache_file(preproc_path, subject, session, task, preload, run=None, proc=None):
    cache_path = get_cache_path(
        preproc_path, subject, session, task, run, proc)
    raw = mne.io.read_raw_fif(cache_path, preload=preload)
    return raw


def get_wakeman_cache_path(preproc_path: str, subject: str, session: str):
    fname = f"sub-{subject}_ses-meg_task-facerecognition_run-{session}_meg.fif"
    subject_cache_path = os.path.join(str(preproc_path), f"sub-{subject}")
    return os.path.join(subject_cache_path, fname)


def read_wakeman_cache_file(preproc_path, subject, session, preload):
    cache_path = get_cache_path(preproc_path, subject, session)
    raw = mne.io.read_raw_fif(cache_path, preload=preload)
    return raw


def get_serialized_files(base_path, key_names: list[str], present_value_combinations: list[list[int]]):
    """

    key_names: list of the names of folders on each level such as "sub" or "ses"
    present_value_combinations: list of lists of indices of the key combinations that are present as folders.

    Returns:
    A list of paths to the serialized files.
    """

    paths = []
    for combination in present_value_combinations:
        if (len(combination) != len(key_names)):
            raise ValueError(
                f"Invalid key combination: {combination} for key names: {key_names}")
        path = os.path.join(base_path, *[
            key_names[i] + "-" + str(combination[i]) for i in range(len(key_names))])
        # get the serialized files named "sample-0.pt" etc.
        # throw an error if the folder does not exist
        if not os.path.exists(path):
            raise ValueError(
                f"Path {path} does not exist. Expected serialized files.")
        sample_paths = glob.glob(os.path.join(path, "sample-*.pt"))
        if len(sample_paths) == 0:
            raise ValueError(
                f"Path {path} does not contain any serialized files.")
        paths.extend(sample_paths)
    return paths


def check_include_and_exclude_ids(include_ids, exclude_ids, all_ids):
    if (set(include_ids) & set(exclude_ids)):
        raise ValueError(
            f"IDs cannot be both included and excluded: {set(include_ids) & set(exclude_ids)}")
    if (set(exclude_ids) - set(all_ids)):
        warnings.warn(
            f"Trying to exclude subjects that are not part of the dataset: {set(exclude_ids) - set(all_ids)}")


def include_exclude_ids(include_ids, exclude_ids, all_ids):
    if include_ids:
        ids = include_ids
    else:
        ids = all_ids
    if exclude_ids:
        ids = [id for id in ids if id not in exclude_ids]
    return ids
