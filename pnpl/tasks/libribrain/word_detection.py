"""
Word Detection/Classification Task for LibriBrain.

Supports multi-class word classification or binary keyword detection.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
import pandas as pd


@dataclass
class WordDetection:
    """
    Word classification/detection task.
    
    Can operate in two modes:
    1. Multi-class: Classify all words
    2. Keyword detection: Binary classification for specific keyword(s)
    
    Args:
        tmin: Start time relative to word onset (seconds). If None, auto-computed.
        tmax: End time relative to word onset (seconds). If None, auto-computed.
        min_word_length: Minimum word length to include
        max_word_length: Maximum word length to include (None for no limit)
        keyword_detection: Keyword(s) for binary detection mode
        negative_buffer: Extra time before word onset when auto-computing tmin
        positive_buffer: Extra time after word end when auto-computing tmax
    """
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    min_word_length: int = 1
    max_word_length: Optional[int] = None
    keyword_detection: Optional[Union[str, list]] = None
    negative_buffer: float = 0.0
    positive_buffer: float = 0.0
    
    # Set after collect_samples
    _words_sorted: list = field(default_factory=list, repr=False)
    _word_to_id: dict = field(default_factory=dict, repr=False)
    _keyword_set: set = field(default_factory=set, repr=False)
    
    def __post_init__(self):
        # Normalize keyword detection to a set
        if isinstance(self.keyword_detection, str):
            self._keyword_set = {self.keyword_detection.lower()}
        elif isinstance(self.keyword_detection, (list, tuple, set)):
            self._keyword_set = {str(kw).lower() for kw in self.keyword_detection}
        else:
            self._keyword_set = set()
        
        # Set default tmin/tmax if not provided
        if self.tmin is None:
            self.tmin = 0.0 - self.negative_buffer
        if self.tmax is None:
            self.tmax = 0.5 + self.positive_buffer
    
    @property
    def is_keyword_mode(self) -> bool:
        """Check if operating in keyword detection mode."""
        return len(self._keyword_set) > 0
    
    @property
    def label_info(self) -> dict:
        """Label metadata."""
        if self.is_keyword_mode:
            return {
                'classes': ['other', 'keyword'],
                'label_to_id': {'other': 0, 'keyword': 1},
                'id_to_label': {0: 'other', 1: 'keyword'},
                'n_classes': 2,
            }
        else:
            return {
                'classes': self._words_sorted,
                'label_to_id': self._word_to_id,
                'id_to_label': {i: w for w, i in self._word_to_id.items()},
                'n_classes': len(self._words_sorted),
            }
    
    def collect_samples(self, dataset) -> list[tuple]:
        """
        Collect word samples from all runs.
        
        Args:
            dataset: LibriBrain dataset instance
            
        Returns:
            List of (subject, session, task, run, onset, word, sent_idx, word_idx) tuples
        """
        samples = []
        all_words = set()
        
        for run_key in dataset.run_keys:
            subject, session, task, run = run_key
            
            # Load word events
            run_samples = self._load_words_for_run(dataset, run_key)
            samples.extend(run_samples)
            
            # Track unique words
            for sample in run_samples:
                word = sample[5]
                if isinstance(word, str):
                    all_words.add(word)
        
        # Build word mappings
        self._words_sorted = sorted(list(all_words))
        self._word_to_id = {w: i for i, w in enumerate(self._words_sorted)}
        
        return samples
    
    def _load_words_for_run(self, dataset, run_key: tuple) -> list[tuple]:
        """Load word events for a single run."""
        subject, session, task, run = run_key
        
        try:
            events_path = dataset.get_events_path(subject, session, task, run)
            if hasattr(dataset, 'ensure_file'):
                events_path = dataset.ensure_file(events_path)
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return []
        
        # Filter for word events
        df = df[df["kind"] == "word"].copy()
        
        # Clean up word column
        if "segment" in df.columns:
            df = df.dropna(subset=["segment", "timemeg"])
            df["segment"] = df["segment"].astype(str).str.strip()
        else:
            return []
        
        # Apply length filters
        if self.min_word_length > 1:
            df = df[df["segment"].str.len() >= self.min_word_length]
        if self.max_word_length is not None:
            df = df[df["segment"].str.len() <= self.max_word_length]
        
        samples = []
        for _, row in df.iterrows():
            word = row["segment"]
            onset = row["timemeg"]
            sent_idx = row.get("sentenceidx", -1)
            word_idx = row.get("wordidx", -1)
            samples.append((subject, session, task, run, onset, word, sent_idx, word_idx))
        
        return samples
    
    def get_label(self, sample: tuple) -> int:
        """
        Extract label ID from sample.
        
        Args:
            sample: (subject, session, task, run, onset, word, sent_idx, word_idx) tuple
            
        Returns:
            Integer label ID
        """
        word = sample[5]
        
        if self.is_keyword_mode:
            # Binary: keyword or not
            if isinstance(word, str) and word.lower() in self._keyword_set:
                return 1
            return 0
        else:
            # Multi-class: word ID
            return self._word_to_id.get(word, 0)

