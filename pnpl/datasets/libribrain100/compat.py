"""
Task-specific wrappers for LibriBrain100.

Mirror the existing ``LibriBrainSpeech`` / ``LibriBrainPhoneme`` /
``LibriBrainWord`` shape so users coming from the old API can switch
without rethinking their constructor calls.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .dataset import LibriBrain100
from .selectors import CorpusArg, PartitionArg, SubjectsArg
from ...tasks.libribrain import (
    PhonemeClassification,
    SpeechDetection,
    WordClassification,
)


class LibriBrain100Speech(LibriBrain100):
    """Speech-vs-silence binary classification on LibriBrain100."""

    def __init__(
        self,
        data_path: str,
        partition: PartitionArg = None,
        subjects: SubjectsArg = "all",
        corpus: CorpusArg = "all",
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys=None,
        exclude_run_keys=None,
        exclude_tasks=None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        oversample_silence_jitter: int = 0,
        preload_files: bool = False,
        stride: Optional[int] = None,
        download: bool = True,
        preload_h5: bool = False,
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
            subjects=subjects,
            corpus=corpus,
            preprocessing_str=preprocessing_str,
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
            preload_h5=preload_h5,
        )


class LibriBrain100Phoneme(LibriBrain100):
    """Multi-class phoneme classification on LibriBrain100."""

    def __init__(
        self,
        data_path: str,
        partition: PartitionArg = None,
        subjects: SubjectsArg = "all",
        corpus: CorpusArg = "all",
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        label_type: str = "phoneme",
        include_run_keys=None,
        exclude_run_keys=None,
        exclude_tasks=None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = False,
        download: bool = True,
        preload_h5: bool = False,
    ):
        task = PhonemeClassification(tmin=tmin, tmax=tmax, label_type=label_type)
        super().__init__(
            data_path=data_path,
            task=task,
            partition=partition,
            subjects=subjects,
            corpus=corpus,
            preprocessing_str=preprocessing_str,
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
            preload_h5=preload_h5,
        )


class LibriBrain100Word(LibriBrain100):
    """Word classification / keyword detection on LibriBrain100."""

    def __init__(
        self,
        data_path: str,
        partition: PartitionArg = None,
        subjects: SubjectsArg = "all",
        corpus: CorpusArg = "all",
        preprocessing_str: str = "bads+headpos+sss+notch+bp+ds",
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        min_word_length: int = 1,
        max_word_length: Optional[int] = None,
        keyword_detection=None,
        negative_buffer: float = 0.0,
        positive_buffer: float = 0.0,
        include_run_keys=None,
        exclude_run_keys=None,
        exclude_tasks=None,
        standardize: bool = True,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        preload_files: bool = False,
        download: bool = True,
        preload_h5: bool = False,
    ):
        task = WordClassification(
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
            subjects=subjects,
            corpus=corpus,
            preprocessing_str=preprocessing_str,
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
            preload_h5=preload_h5,
        )


__all__ = [
    "LibriBrain100Speech",
    "LibriBrain100Phoneme",
    "LibriBrain100Word",
]
