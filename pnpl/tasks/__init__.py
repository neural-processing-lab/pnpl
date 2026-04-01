"""
Tasks module for PNPL datasets.

Tasks define how samples are collected from datasets and how labels are extracted.
They are composable and can be easily extended for custom use cases.
"""

try:
    from .._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

from .base import TaskProtocol

# LibriBrain tasks
from .libribrain import (
    SpeechDetection,
    PhonemeClassification,
    WordDetection,
)

__all__ = [
    # Protocol
    "TaskProtocol",
    # LibriBrain
    "SpeechDetection",
    "PhonemeClassification",
    "WordDetection",
]
