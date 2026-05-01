"""
LibriBrain-specific tasks.

These tasks are designed for the LibriBrain dataset which contains:
- Continuous MEG recordings
- Event annotations in TSV files (words, phonemes, silence)
"""

try:
    from ..._namespace import extend_overlay_path as _extend_overlay_path

    __path__ = _extend_overlay_path(__path__, __name__)
except Exception:
    pass

from .speech_detection import SpeechDetection
from .phoneme_classification import PhonemeClassification
from .word_detection import WordDetection

__all__ = [
    "SpeechDetection",
    "PhonemeClassification",
    "WordDetection",
]
