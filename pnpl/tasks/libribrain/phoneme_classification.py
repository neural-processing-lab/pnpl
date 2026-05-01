"""
Phoneme Classification Task for LibriBrain.

Multi-class classification of phonemes from MEG data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

from ...datasets.libribrain2025.constants import (
    PHONATION_BY_PHONEME,
    PHONEME_LABELS_SORTED,
)


@dataclass
class PhonemeClassification:
    """
    Multi-class phoneme classification task.
    
    Each sample is aligned to a phoneme onset in the events file.
    
    Args:
        tmin: Start time relative to phoneme onset (seconds)
        tmax: End time relative to phoneme onset (seconds)
        label_type: 'phoneme' for multi-class, 'voicing' for binary (voiced/unvoiced)
        exclude_phonemes: List of phonemes to exclude
    """
    tmin: float = 0.0
    tmax: float = 0.5
    label_type: str = "phoneme"  # 'phoneme' or 'voicing'
    exclude_phonemes: list = field(default_factory=list)
    
    # Set after collect_samples
    _phonemes_sorted: list = field(default_factory=list, repr=False)
    _phoneme_to_id: dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        if self.label_type not in ("phoneme", "voicing"):
            raise ValueError(f"label_type must be 'phoneme' or 'voicing', got {self.label_type}")
    
    @property
    def label_info(self) -> dict:
        """Label metadata."""
        if self.label_type == "voicing":
            return {
                'classes': ['uv', 'v'],  # unvoiced, voiced
                'label_to_id': {'uv': 0, 'v': 1},
                'id_to_label': {0: 'uv', 1: 'v'},
                'n_classes': 2,
            }
        else:
            return {
                'classes': self._phonemes_sorted,
                'label_to_id': self._phoneme_to_id,
                'id_to_label': {i: p for p, i in self._phoneme_to_id.items()},
                'n_classes': len(self._phonemes_sorted),
            }
    
    def collect_samples(self, dataset) -> list[tuple]:
        """
        Collect phoneme samples from all runs.
        
        Args:
            dataset: LibriBrain dataset instance
            
        Returns:
            List of (subject, session, task, run, onset, phoneme_label) tuples
        """
        samples = []
        exclude_set = set(self.exclude_phonemes)
        allowed_phonemes = [p for p in PHONEME_LABELS_SORTED if p not in exclude_set]
        
        for run_key in dataset.run_keys:
            # Load phoneme events
            run_samples = self._load_phonemes_for_run(dataset, run_key)
            samples.extend(run_samples)
        
        # Keep label IDs stable across train/validation/test splits.
        self._phonemes_sorted = allowed_phonemes
        self._phoneme_to_id = {p: i for i, p in enumerate(self._phonemes_sorted)}
        
        return samples
    
    def _load_phonemes_for_run(self, dataset, run_key: tuple) -> list[tuple]:
        """Load phoneme events for a single run."""
        subject, session, task, run = run_key
        
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            if hasattr(dataset, 'ensure_file'):
                events_path = dataset.ensure_file(events_path)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []
        
        # Filter for phoneme events
        df = df[df["kind"] == "phoneme"]
        df = df[df["segment"] != "oov_S"]  # Out of vocabulary
        df = df[df["segment"] != "sil"]    # Silence
        df = df[df["segment"].str.split("_").str[0].isin(PHONEME_LABELS_SORTED)]
        
        # Apply exclusions
        if self.exclude_phonemes:
            exclude_set = set(self.exclude_phonemes)
            df = df[~df["segment"].str.split("_").str[0].isin(exclude_set)]
        
        samples = []
        for _, row in df.iterrows():
            phoneme = row["segment"]
            onset = row["timemeg"]
            samples.append((subject, session, task, run, onset, phoneme))
        
        return samples
    
    def get_label(self, sample: tuple) -> int:
        """
        Extract label ID from sample.
        
        Args:
            sample: (subject, session, task, run, onset, phoneme_full) tuple
            
        Returns:
            Integer label ID
        """
        phoneme_full = sample[5]
        phoneme = phoneme_full.split("_")[0]  # Remove position marker
        
        if self.label_type == "voicing":
            voicing = PHONATION_BY_PHONEME.get(phoneme, 'uv')
            return 0 if voicing == 'uv' else 1
        else:
            return self._phoneme_to_id.get(phoneme, 0)
