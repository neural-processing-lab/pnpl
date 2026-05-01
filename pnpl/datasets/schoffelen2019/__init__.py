"""Schöffelen et al. (2019) MOUS MEG dataset module."""

from .dataset import Schoffelen2019
from .constants import (
    CHANNELS,
    RADBOUD_DATASET_URL,
    SUBJECTS,
    TASKS,
    is_task_for_subject,
)

__all__ = [
    "Schoffelen2019",
    "CHANNELS",
    "RADBOUD_DATASET_URL",
    "SUBJECTS",
    "TASKS",
    "is_task_for_subject",
]
