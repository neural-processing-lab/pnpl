"""MEG-MASC (Gwilliams et al., 2022) dataset module."""

from .dataset import Gwilliams2022
from .constants import (
    OSF_PROJECT_ID,
    OSF_PROJECT_FALLBACKS,
    SESSIONS,
    SUBJECTS,
    TASKS,
    TASK_STORIES,
)

__all__ = [
    "Gwilliams2022",
    "OSF_PROJECT_ID",
    "OSF_PROJECT_FALLBACKS",
    "SESSIONS",
    "SUBJECTS",
    "TASKS",
    "TASK_STORIES",
]
