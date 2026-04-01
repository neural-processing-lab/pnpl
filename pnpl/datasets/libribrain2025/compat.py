"""
Wrapper classes for LibriBrain.

These classes provide specialized dataset entry points built on top of the
task-based ``LibriBrain`` dataset.
"""

import numpy as np
from typing import Optional

from .dataset import LibriBrain
from ...tasks.libribrain import SpeechDetection, PhonemeClassification, WordDetection


class LibriBrainSpeech(LibriBrain):
    """
    Speech detection dataset wrapper.
    
    Binary classification of speech vs silence segments.
    
    Args:
        data_path: Path to store/load the dataset
        partition: train/validation/test split
        preprocessing_str: Preprocessing string for filenames
        tmin: Start time relative to window position
        tmax: End time relative to window position
        include_run_keys: Specific runs to include
        exclude_run_keys: Specific runs to exclude
        exclude_tasks: Task names to exclude
        standardize: Whether to z-score normalize
        clipping_boundary: Clip values to [-boundary, boundary]
        channel_means: Pre-computed channel means
        channel_stds: Pre-computed channel stds
        include_info: Include metadata in samples
        oversample_silence_jitter: Stride for oversampling silence
        preload_files: Eagerly download files
        stride: Custom stride for sliding window
        download: Enable HuggingFace downloads
    """
    
    def __init__(
        self,
        data_path: str,
        partition: Optional[str] = None,
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: list = None,
        exclude_run_keys: list = None,
        exclude_tasks: list = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        oversample_silence_jitter: int = 0,
        preload_files: bool = True,
        stride: Optional[int] = None,
        download: bool = True,
    ):
        task = SpeechDetection(
            tmin=tmin,
            tmax=tmax,
            stride=stride,
            oversample_silence_jitter=oversample_silence_jitter,
        )
        
        super().__init__(
            data_path=data_path,
            task=task,
            partition=partition,
            preprocessing=preprocessing_str,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            exclude_tasks=exclude_tasks,
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
            include_info=include_info,
            preload_files=preload_files,
            download=download,
        )


class LibriBrainPhoneme(LibriBrain):
    """
    Phoneme classification dataset wrapper.
    
    Multi-class classification of phonemes.
    
    Args:
        data_path: Path to store/load the dataset
        partition: train/validation/test split
        label_type: 'phoneme' or 'voicing'
        preprocessing_str: Preprocessing string for filenames
        tmin: Start time relative to phoneme onset
        tmax: End time relative to phoneme onset
        include_run_keys: Specific runs to include
        exclude_run_keys: Specific runs to exclude
        exclude_tasks: Task names to exclude
        standardize: Whether to z-score normalize
        clipping_boundary: Clip values to [-boundary, boundary]
        channel_means: Pre-computed channel means
        channel_stds: Pre-computed channel stds
        include_info: Include metadata in samples
        preload_files: Eagerly download files
        download: Enable HuggingFace downloads
    """
    
    def __init__(
        self,
        data_path: str,
        partition: Optional[str] = None,
        label_type: str = "phoneme",
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: list = None,
        exclude_run_keys: list = None,
        exclude_tasks: list = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = True,
        download: bool = True,
    ):
        task = PhonemeClassification(
            tmin=tmin,
            tmax=tmax,
            label_type=label_type,
        )
        
        super().__init__(
            data_path=data_path,
            task=task,
            partition=partition,
            preprocessing=preprocessing_str,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            exclude_tasks=exclude_tasks,
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
            include_info=include_info,
            preload_files=preload_files,
            download=download,
        )
        
        # Expose phoneme mappings for compatibility
        self.phonemes_sorted = task._phonemes_sorted
        self.phoneme_to_id = task._phoneme_to_id
        self.id_to_phoneme = task._phonemes_sorted
        self.labels_sorted = task.label_info['classes']
        self.label_to_id = task.label_info['label_to_id']


class LibriBrainWord(LibriBrain):
    """
    Word classification dataset wrapper.
    
    Multi-class word classification or binary keyword detection.
    
    Args:
        data_path: Path to store/load the dataset
        partition: train/validation/test split
        preprocessing_str: Preprocessing string for filenames
        tmin: Start time relative to word onset
        tmax: End time relative to word onset
        include_run_keys: Specific runs to include
        exclude_run_keys: Specific runs to exclude
        exclude_tasks: Task names to exclude
        standardize: Whether to z-score normalize
        clipping_boundary: Clip values to [-boundary, boundary]
        channel_means: Pre-computed channel means
        channel_stds: Pre-computed channel stds
        include_info: Include metadata in samples
        preload_files: Eagerly download files
        download: Enable HuggingFace downloads
        min_word_length: Minimum word length to include
        max_word_length: Maximum word length to include
        keyword_detection: Keyword(s) for binary detection
        negative_buffer: Extra time before word onset
        positive_buffer: Extra time after word end
    """
    
    def __init__(
        self,
        data_path: str,
        partition: Optional[str] = None,
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        include_run_keys: list = None,
        exclude_run_keys: list = None,
        exclude_tasks: list = None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = True,
        download: bool = True,
        min_word_length: int = 1,
        max_word_length: Optional[int] = None,
        keyword_detection: Optional[str] = None,
        negative_buffer: float = 0.0,
        positive_buffer: float = 0.0,
    ):
        task = WordDetection(
            tmin=tmin,
            tmax=tmax,
            min_word_length=min_word_length,
            max_word_length=max_word_length,
            keyword_detection=keyword_detection,
            negative_buffer=negative_buffer,
            positive_buffer=positive_buffer,
        )
        
        super().__init__(
            data_path=data_path,
            task=task,
            partition=partition,
            preprocessing=preprocessing_str,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            exclude_tasks=exclude_tasks,
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
            include_info=include_info,
            preload_files=preload_files,
            download=download,
        )
        
        # Expose word mappings for compatibility
        self.words_sorted = task._words_sorted
        self.word_to_id = task._word_to_id
        self.id_to_word = task._words_sorted
        self.labels_sorted = task.label_info['classes']
        self.label_to_id = task.label_info['label_to_id']
