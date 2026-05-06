"""Tasks for the MEG-MASC (Gwilliams et al., 2022) dataset."""

from .phoneme_classification import PhonemeClassification
from .word_classification import WordClassification, WordDetection

__all__ = ["PhonemeClassification", "WordClassification", "WordDetection"]
