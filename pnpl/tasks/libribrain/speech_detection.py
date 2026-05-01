"""
Speech Detection Task for LibriBrain.

Binary classification: speech vs silence segments.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass
class SpeechDetection:
    """
    Binary speech vs silence classification task.
    
    This task slides a time window across continuous MEG data and labels
    each window based on whether it contains speech or silence.
    
    Args:
        tmin: Start time of sample window relative to sliding position (seconds)
        tmax: End time of sample window relative to sliding position (seconds)
        stride: Step size for sliding window (samples). If None, uses window size.
        oversample_silence_jitter: If > 0, oversample silence segments with this stride
    """
    tmin: float = 0.0
    tmax: float = 0.5
    stride: Optional[int] = None
    oversample_silence_jitter: int = 0
    
    # Internal state (set after collect_samples)
    _classes: list = field(default_factory=lambda: ["silence", "speech"], repr=False)
    
    @property
    def label_info(self) -> dict:
        """Label metadata for binary classification."""
        return {
            'classes': self._classes,
            'label_to_id': {'silence': 0, 'speech': 1},
            'id_to_label': {0: 'silence', 1: 'speech'},
            'n_classes': 2,
        }
    
    def collect_samples(self, dataset) -> list[tuple]:
        """
        Collect speech/silence samples from all runs.
        
        Args:
            dataset: LibriBrain dataset instance
            
        Returns:
            List of (subject, session, task, run, onset, label_array) tuples
        """
        samples = []
        sfreq = dataset.sfreq
        time_window_samples = int((self.tmax - self.tmin) * sfreq)
        stride = self.stride if self.stride is not None else time_window_samples
        
        for run_key in dataset.run_keys:
            subject, session, task, run = run_key
            
            # Get speech labels for this run
            speech_labels = self._get_speech_labels_for_run(dataset, run_key)
            if speech_labels is None:
                continue
            
            if self.oversample_silence_jitter > 0:
                run_samples = self._collect_oversampled(
                    subject, session, task, run,
                    speech_labels, sfreq, time_window_samples,
                    self.oversample_silence_jitter,
                )
            else:
                run_samples = self._collect_windowed(
                    subject, session, task, run,
                    speech_labels, sfreq, time_window_samples, stride,
                )
            
            samples.extend(run_samples)
        
        return samples
    
    def _get_speech_labels_for_run(self, dataset, run_key: tuple) -> Optional[np.ndarray]:
        """
        Build speech/silence label array for a run.
        
        Returns:
            Array where 1=speech, 0=silence, indexed by sample number
        """
        subject, session, task, run = run_key
        sfreq = dataset.sfreq
        
        # Load events
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            if hasattr(dataset, 'ensure_file'):
                events_path = dataset.ensure_file(events_path)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return None
        
        # Convert times to samples
        df['timemeg_samples'] = (pd.to_numeric(df['timemeg'], errors='coerce') * sfreq).astype(int)
        df['duration_samples'] = (pd.to_numeric(df['duration'], errors='coerce') * sfreq).astype(int)
        
        # Filter for silence and word entries
        silence_df = df[df['kind'] == 'silence']
        words_df = df[df['kind'] == 'word']
        
        if silence_df.empty or words_df.empty:
            return None
        
        # Determine array size
        max_word = (words_df['timemeg_samples'] + words_df['duration_samples']).max()
        max_silence = (silence_df['timemeg_samples'] + silence_df['duration_samples']).max()
        array_size = max(max_word, max_silence) + 1
        
        # Initialize with speech (1)
        speech_labels = np.ones(array_size, dtype=np.int32)
        
        # Set pre-annotation period to silence
        min_word = words_df['timemeg_samples'].min()
        min_silence = silence_df['timemeg_samples'].min()
        min_sample = min(min_word, min_silence)
        speech_labels[:min_sample] = 0
        
        # Mark silence segments
        for _, row in silence_df.iterrows():
            start = row['timemeg_samples']
            duration = row['duration_samples']
            if not np.isnan(start) and not np.isnan(duration):
                end = int(start + duration)
                speech_labels[int(start):end] = 0
        
        return speech_labels
    
    def _collect_windowed(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        speech_labels: np.ndarray,
        sfreq: float,
        window_size: int,
        stride: int,
    ) -> list[tuple]:
        """Collect samples with fixed stride."""
        samples = []
        
        for i in range(0, len(speech_labels) - window_size, stride):
            label_segment = speech_labels[i:i + window_size]
            onset = i / sfreq
            samples.append((subject, session, task, run, onset, label_segment))
        
        return samples
    
    def _collect_oversampled(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        speech_labels: np.ndarray,
        sfreq: float,
        window_size: int,
        silence_jitter: int,
    ) -> list[tuple]:
        """
        Collect samples with oversampling of silence regions.
        
        Uses normal stride for speech, smaller stride around silence.
        """
        samples = []
        i = 0
        jitter_active = False
        step_size = window_size
        
        while i < len(speech_labels) - window_size:
            label_segment = speech_labels[i:i + window_size]
            
            # Check if we hit silence
            if label_segment.sum() < window_size and not jitter_active:
                jitter_active = True
                first_zero = np.argmax(label_segment == 0)
                i = i - (window_size - first_zero - 1)
                step_size = silence_jitter
                continue
            
            # Check if we're back to all speech
            if label_segment.sum() == window_size and jitter_active:
                jitter_active = False
                step_size = window_size
            
            onset = i / sfreq
            
            # Add sample (with filtering for transition regions)
            if jitter_active:
                ratio = label_segment.sum() / len(label_segment)
                if 0.3 < ratio < 0.5 or label_segment.sum() == 0:
                    samples.append((subject, session, task, run, onset, label_segment))
            else:
                samples.append((subject, session, task, run, onset, label_segment))
            
            i += step_size
        
        return samples
    
    def get_label(self, sample: tuple) -> np.ndarray:
        """
        Extract label array from sample.
        
        Args:
            sample: (subject, session, task, run, onset, label_array) tuple
            
        Returns:
            Label array (0=silence, 1=speech) for each time point
        """
        return sample[5]

