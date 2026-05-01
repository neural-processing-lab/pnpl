"""Armeni et al. (2022) MEG dataset module."""

from .dataset import Armeni2022
from .constants import RADBOUD_DATASET_URL, SESSIONS, SUBJECTS, TASKS

__all__ = ["Armeni2022", "RADBOUD_DATASET_URL", "SESSIONS", "SUBJECTS", "TASKS"]
