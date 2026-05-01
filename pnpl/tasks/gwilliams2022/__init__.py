"""Tasks for the MEG-MASC (Gwilliams et al., 2022) dataset."""

from .phoneme_classification import PhonemeClassification
from .word_detection import WordDetection

__all__ = ["PhonemeClassification", "WordDetection"]
