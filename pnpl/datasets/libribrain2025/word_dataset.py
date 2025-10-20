import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from pnpl.datasets.libribrain2025.base import LibriBrainBase
from pnpl.datasets.libribrain2025.constants import RUN_KEYS, VALIDATION_RUN_KEYS, TEST_RUN_KEYS


class LibriBrainWord(LibriBrainBase):

    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
        preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
        tmin: float | None = None,
        tmax: float | None = None,
        include_run_keys: list[str] = [],
        exclude_run_keys: list[str] = [],
        exclude_tasks: list[str] = [],
        standardize: bool = True,
        clipping_boundary: float | None = 10,
        channel_means: np.ndarray | None = None,
        channel_stds: np.ndarray | None = None,
        include_info: bool = False,
        preload_files: bool = True,
        download: bool = True,
        min_word_length: int = 1,
        max_word_length: int | None = None,
        keyword_detection: str | list[str] | None = None,
        negative_buffer: float = 0.0,
        positive_buffer: float = 0.0,
    ):
        """
        LibriBrain word classification dataset.

        This dataset provides MEG data aligned to word onsets for word classification/detection tasks.
        Each sample contains MEG data from tmin to tmax seconds relative to a word onset.

        Args:
            data_path: Path where you wish to store the dataset. The local dataset structure 
                      will follow the same BIDS-like structure as the HuggingFace repo:
                      ```
                      data_path/
                      ├── {task}/                    # e.g., "Sherlock1"
                      │   └── derivatives/
                      │       ├── serialised/       # MEG data files
                      │       │   └── sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing_str}_meg.h5
                      │       └── events/            # Event timing files  
                      │           └── sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv
                      ```
            partition: Convenient shortcut to specify train/validation/test split. Use "train", 
                      "validation", or "test". Instead of specifying run keys manually, you can use:
                      - partition="train": All runs except validation and test
                      - partition="validation": ('0', '11', 'Sherlock1', '2') 
                      - partition="test": ('0', '12', 'Sherlock1', '2')
                      Note on keyword-aware partitions: If keyword_detection is provided and you
                      pass partition (with no include_run_keys/exclude_run_keys), the dataset will
                      select validation/test runs that contain the keyword(s) by scanning only the
                      events files. It prefers the default validation/test splits; if absent, it
                      chooses alternatives and caches the choice at data_path/_cache/
                      keyword_detection_splits.json for reproducibility. With download=False, only
                      locally present events.tsv files are considered and an error is raised if no
                      matching runs are found. This behaviour does not apply when explicit
                      include_run_keys/exclude_run_keys are provided.
            preprocessing_str: By default, we expect files with preprocessing string 
                             "bads+headpos+sss+notch+bp+ds". This indicates the preprocessing steps:
                             bads+headpos+sss+notch+bp+ds means the data has been processed for 
                             bad channel removal, head position adjustment, signal-space separation, 
                             notch filtering, bandpass filtering, and downsampling.
            tmin: Start time (s) relative to word onset. If None, a default is
                 auto-generated; when auto-generated, negative_buffer is subtracted.
            tmax: End time (s) relative to word onset. If None, a default is
                 auto-generated; when auto-generated, positive_buffer is added.
                 The number of timepoints per sample = int((tmax - tmin) * sfreq) where sfreq=250Hz.
            include_run_keys: List of specific sessions to include. Format per session: 
                            ('0', '1', 'Sherlock1', '1') = Subject 0, Session 1, Task Sherlock1, Run 1.
                            You can see all valid run keys by importing RUN_KEYS from 
                            pnpl.datasets.libribrain2025.constants.
            exclude_run_keys: List of sessions to exclude (same format as include_run_keys).
            exclude_tasks: List of task names to exclude (e.g., ['Sherlock1']).
            standardize: Whether to z-score normalize each channel's MEG data using mean and std 
                        computed across all included runs.
                        Formula: normalized_data[channel] = (raw_data[channel] - channel_mean[channel]) / channel_std[channel]
            clipping_boundary: If specified, clips all values to [-clipping_boundary, clipping_boundary]. 
                             This can help with outliers. Set to None for no clipping.
            channel_means: Pre-computed channel means for standardization. If provided along with 
                          channel_stds, these will be used instead of computing from the dataset.
            channel_stds: Pre-computed channel standard deviations for standardization.
            include_info: Whether to include additional info dict in each sample containing dataset name, 
                         subject, session, task, run, onset time, word text, sentence index, and word index.
            preload_files: Whether to "eagerly" download all dataset files from HuggingFace when 
                          the dataset object is created (True) or "lazily" download files on demand (False). 
                          We recommend leaving this as True unless you have a specific reason not to.
            download: Whether to download files from HuggingFace if not found locally (True) or 
                     throw an error if files are missing locally (False).
            min_word_length: Minimum length of words to include (filters out very short words).
            max_word_length: Maximum length of words to include (filters out very long words). 
                           Set to None for no maximum length filtering.
            keyword_detection: If specified, converts to binary classification mode for detecting 
                             a keyword or a list of keywords. Labels become 0 (not keyword) or 
                             1 (keyword). Case-insensitive matching. You may pass a single string 
                             or a list of strings; when a list is provided, any occurrence of the 
                             listed words is treated as a positive label.
            negative_buffer: Extra seconds to prepend when tmin is auto-selected based on longest keyword duration.
            positive_buffer: Extra seconds to append when tmax is auto-selected based on longest keyword duration.

        Returns:
            Data samples with shape (channels, time) where channels=306 MEG channels.
            Labels are integers corresponding to word classes (multi-class) or binary labels 
            (0/1) if keyword_detection is specified.
        """
        # Prepare keyword set (multi-keyword support)
        self.keyword_set = None
        self.keyword_signature = None
        if isinstance(keyword_detection, str):
            self.keyword_set = {keyword_detection.lower()}
        elif isinstance(keyword_detection, (list, tuple, set)) and len(keyword_detection) > 0:
            self.keyword_set = {str(kw).lower() for kw in keyword_detection}
        else:
            self.keyword_set = None
        if self.keyword_set:
            self.keyword_signature = ",".join(sorted(self.keyword_set))

        # Ensure flags used before super().__init__ are available
        self.download = download
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.negative_buffer = float(negative_buffer)
        self.positive_buffer = float(positive_buffer)
        # Keep original for reference; we primarily use self.keyword_set
        if isinstance(keyword_detection, str):
            self.keyword_detection = keyword_detection.lower()
        elif isinstance(keyword_detection, (list, tuple, set)) and len(keyword_detection) > 0:
            # For lists, keep a readable token
            self.keyword_detection = "keywords"
        else:
            self.keyword_detection = None

        # If keyword detection is enabled and a partition is requested, choose keyword-aware splits
        effective_partition = partition
        effective_include_run_keys = include_run_keys
        effective_exclude_run_keys = exclude_run_keys
        if self.keyword_set is not None and partition in {"train", "validation", "test"} and not include_run_keys and not exclude_run_keys:
            # Dynamically pick validation/test runs that actually contain the keyword(s)
            chosen_val, chosen_test = self._choose_keyword_aware_splits(
                data_path=data_path,
                exclude_tasks=exclude_tasks,
            )
            # Convert partition intent into explicit run key selection for reproducibility
            if partition == "validation":
                effective_include_run_keys = [tuple(chosen_val)]
                effective_partition = None
            elif partition == "test":
                effective_include_run_keys = [tuple(chosen_test)]
                effective_partition = None
            elif partition == "train":
                effective_exclude_run_keys = [tuple(chosen_val), tuple(chosen_test)]
                effective_partition = None

        # Optional: scan keyword durations globally before base init to allow consistent window defaults
        scanned_max_kw_duration = None
        scanned_max_kw_onset = None
        scanned_max_kw_word = None
        scanned_source = None
        if self.keyword_set is not None:
            dur, ons, w, src = self._get_or_compute_global_keyword_duration(data_path=data_path, exclude_tasks=exclude_tasks)
            if dur is not None:
                scanned_max_kw_duration = dur
                scanned_max_kw_onset = ons
                scanned_max_kw_word = w
                scanned_source = src
                try:
                    src_str = f"source={scanned_source}" if scanned_source else "source=unknown"
                    part_str = f", partition={partition}" if partition is not None else ""
                    # We will fill in window values after inference below; print after setting defaults
                except Exception:
                    pass

                # If tmin/tmax not explicitly provided, set dynamic defaults consistently
                if tmin is None:
                    tmin = 0 - self.negative_buffer
                if tmax is None:
                    tmax = scanned_max_kw_duration + self.positive_buffer

                # If both provided and total window shorter than max keyword duration, warn
                if tmin is not None and tmax is not None and (tmax - tmin) < scanned_max_kw_duration:
                    warnings.warn(
                        f"Provided window ({tmax - tmin:.3f}s) is shorter than longest keyword duration ({scanned_max_kw_duration:.3f}s). Consider increasing tmax.")
                # After determining window, emit the informative note
                try:
                    src_str = f"source={scanned_source}" if scanned_source else "source=unknown"
                    part_str = f", partition={partition}" if partition is not None else ""
                    word_str = f" ({scanned_max_kw_word})" if scanned_max_kw_word is not None else ""
                    print(f"longest keyword {scanned_max_kw_duration:.3f}s detected at {scanned_max_kw_onset:.3f}s{word_str} [{src_str}{part_str}], window: tmin={tmin:.3f}, tmax={tmax:.3f}")
                except Exception:
                    pass
            else:
                # No durations found; keep user/default values as-is
                if tmin is None:
                    tmin = 0.0 - self.negative_buffer
                if tmax is None:
                    tmax = 0.5 + self.positive_buffer
        else:
            # Not in keyword detection mode; ensure concrete values
            if tmin is None:
                tmin = 0.0 - self.negative_buffer
            if tmax is None:
                tmax = 0.5 + self.positive_buffer

        super().__init__(
            data_path=data_path,
            partition=effective_partition,
            preprocessing_str=preprocessing_str,
            tmin=tmin,
            tmax=tmax,
            include_run_keys=effective_include_run_keys,
            exclude_run_keys=effective_exclude_run_keys,
            exclude_tasks=exclude_tasks,
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
            include_info=include_info,
            preload_files=preload_files,
            download=download,
        )
        
        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")

        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.samples = []
        run_keys_missing = []
        self.run_keys = []
        
        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                words, onsets, sentence_indices, word_indices = self.load_words_from_tsv(
                    subject, session, task, run)
                for word, onset, sent_idx, word_idx in zip(words, onsets, sentence_indices, word_indices):
                    sample = (subject, session, task, run, onset, word, sent_idx, word_idx)
                    self.samples.append(sample)
                self.run_keys.append(run_key)
            except FileNotFoundError:
                run_keys_missing.append(run_key)
                warnings.warn(
                    f"File not found for run key {run_key}. Skipping")
                continue

        if len(run_keys_missing) > 0:
            warnings.warn(
                f"Run keys {run_keys_missing} not found in dataset. Present run keys: {self.run_keys}")

        if len(self.samples) == 0:
            raise ValueError("No samples found.")

        self.words_sorted = self._get_unique_word_labels()
        self.word_to_id = {word: i for i, word in enumerate(self.words_sorted)}
        self.id_to_word = self.words_sorted
        
        # Configure labels based on mode
        if self.keyword_set is not None:
            # Binary classification mode (single or multiple keywords)
            label_name = self.keyword_detection if isinstance(self.keyword_detection, str) else "keyword"
            self.labels_sorted = ["other", label_name]
            self.label_to_id = {"other": 0, label_name: 1}
        else:
            # Multi-class classification mode
            self.labels_sorted = self.words_sorted
            self.label_to_id = self.word_to_id
        
        if (self.standardize and channel_means is None and channel_stds is None):
            self._calculate_standardization_params()
        elif (self.standardize and (channel_means is not None and channel_stds is not None)):
            self.channel_means = channel_means
            self.channel_stds = channel_stds
            self.broadcasted_stds = np.tile(
                self.channel_stds, (self.points_per_sample, 1)).T
            self.broadcasted_means = np.tile(
                self.channel_means, (self.points_per_sample, 1)).T

    def _events_path_from_run_key(self, data_path: str, run_key: tuple[str, str, str, str]) -> str:
        subject, session, task, run = run_key
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        return os.path.join(data_path, task, "derivatives", "events", fname)

    def _load_keyword_cache(self, data_path: str) -> dict:
        cache_dir = os.path.join(data_path, "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "keyword_detection_splits.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {"keywords": {}}
        return {"keywords": {}}

    def _save_keyword_cache(self, data_path: str, cache: dict) -> None:
        cache_dir = os.path.join(data_path, "_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "keyword_detection_splits.json")
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cache, f)
        os.replace(tmp_path, cache_path)

    def _keyword_counts_in_run(self, data_path: str, run_key: tuple[str, str, str, str]) -> tuple[int, int]:
        """Return (pos_count, total_words) for the given run.

        Applies the same word-length filtering as the dataset.
        Respects self.download: when False, only uses local files and returns (0,0) if missing.
        """
        events_path = self._events_path_from_run_key(data_path, run_key)
        if self.download:
            LibriBrainBase.ensure_file_download(events_path, data_path)
        else:
            if not os.path.exists(events_path):
                return 0, 0
        try:
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return 0, 0
        if "kind" in df.columns:
            df = df[df["kind"] == "word"].copy()
        if "segment" not in df.columns:
            return 0, 0
        # Normalise and apply dataset word-length filtering
        seg = df["segment"].astype(str).str.strip()
        if self.min_word_length > 1:
            seg = seg[seg.str.len() >= self.min_word_length]
        if self.max_word_length is not None:
            seg = seg[seg.str.len() <= self.max_word_length]
        total = int(seg.shape[0])
        pos = int(seg.str.lower().isin(self.keyword_set).sum()) if total > 0 else 0
        return pos, total

    def _keyword_presence_in_run(self, data_path: str, run_key: tuple[str, str, str, str]) -> bool:
        pos, total = self._keyword_counts_in_run(data_path, run_key)
        return pos > 0

    def _choose_keyword_aware_splits(self, data_path: str, exclude_tasks: list[str]):
        assert self.keyword_set is not None
        signature = self.keyword_signature
        cache = self._load_keyword_cache(data_path)
        cache.setdefault("keywords", {})
        kw_entry = cache["keywords"].setdefault(signature, {"presence": {}, "splits": {}})

        def key_to_str(rk):
            return "|".join(list(rk))

        kw_entry.setdefault("counts", {})

        def get_counts(rk):
            rk = tuple(rk)
            rk_key = key_to_str(rk)
            if rk_key not in kw_entry["counts"]:
                pos, total = self._keyword_counts_in_run(data_path, rk)
                kw_entry["counts"][rk_key] = {"pos": int(pos), "total": int(total)}
            return kw_entry["counts"][rk_key]["pos"], kw_entry["counts"][rk_key]["total"]

        def get_presence(rk):
            pos, total = get_counts(rk)
            present = pos > 0
            rk_key = key_to_str(tuple(rk))
            kw_entry["presence"][rk_key] = present
            return present

        # Build candidate pools respecting exclude_tasks
        def not_excluded(rk):
            return rk[2] not in set(exclude_tasks)

        val_defaults = [tuple(rk) for rk in VALIDATION_RUN_KEYS if not_excluded(tuple(rk))]
        test_defaults = [tuple(rk) for rk in TEST_RUN_KEYS if not_excluded(tuple(rk))]
        all_candidates = [tuple(rk) for rk in RUN_KEYS if not_excluded(tuple(rk))]

        # Build candidate list where keyword occurs at least once
        candidates = [rk for rk in all_candidates if get_presence(rk)]
        if not candidates:
            if not self.download:
                raise ValueError("Keyword(s) not found in any locally available runs. Either enable download, provide explicit include_run_keys, or ensure events files are present.")
            raise ValueError("Keyword(s) not found in any available runs.")

        # Pre-compute counts for all candidates and for all runs
        counts_map = {}
        for rk in all_candidates:
            pos, total = get_counts(rk)
            counts_map[tuple(rk)] = (pos, total)

        # Select the highest-prevalence runs for validation and test
        def rate(pos, total):
            return (pos / total) if total > 0 else 0.0

        val_set_defaults = set(val_defaults)
        test_set_defaults = set(test_defaults)

        # Score candidates by prevalence (pos/total), then by absolute positives, then prefer defaults
        scored = []  # (rate, pos, total, rk)
        for rk in candidates:
            p, t = counts_map[rk]
            scored.append((rate(p, t), p, t, rk))

        if not scored:
            if not self.download:
                raise ValueError("Keyword(s) not found in any locally available runs. Either enable download, provide explicit include_run_keys, or ensure events files are present.")
            raise ValueError("Keyword(s) not found in any available runs.")

        scored_val = sorted(scored, key=lambda x: (-x[0], -x[1], 0 if x[3] in val_set_defaults else 1))
        chosen_val = scored_val[0][3]

        # For test, avoid using the same run; prefer test-default membership on ties
        scored_test_pool = [s for s in scored if s[3] != chosen_val]
        if not scored_test_pool:
            if not self.download:
                raise ValueError("Keyword(s) not found in any locally available test runs. Either enable download, provide explicit include_run_keys, or ensure events files are present.")
            raise ValueError("Keyword(s) not found in any available runs for test.")
        scored_test = sorted(scored_test_pool, key=lambda x: (-x[0], -x[1], 0 if x[3] in test_set_defaults else 1))
        chosen_test = scored_test[0][3]

        # Persist splits for reproducibility (while still recomputing if exclude_tasks change)
        kw_entry["splits"] = {"validation": list(chosen_val), "test": list(chosen_test)}
        cache["keywords"][signature] = kw_entry
        self._save_keyword_cache(data_path, cache)

        # Informative logging of chosen validation/test splits (robust + always print)
        try:
            def rk_human(rk) -> str:
                return f"sub-{rk[0]} ses-{rk[1]} task-{rk[2]} run-{rk[3]}"

            # Compute pos/neg totals for selected runs (fallback to 0 on error)
            try:
                val_pos, val_tot = get_counts(tuple(chosen_val))
            except Exception:
                val_pos, val_tot = 0, 0
            val_neg = max(0, int(val_tot) - int(val_pos))

            try:
                test_pos, test_tot = get_counts(tuple(chosen_test))
            except Exception:
                test_pos, test_tot = 0, 0
            test_neg = max(0, int(test_tot) - int(test_pos))

            val_tag = "default" if tuple(chosen_val) in set(val_defaults) else "non-default"
            test_tag = "default" if tuple(chosen_test) in set(test_defaults) else "non-default"

            print(f"validation split: {rk_human(tuple(chosen_val))} [{val_tag}] [pos={val_pos}, neg={val_neg}, total={val_tot}]")
            print(f"test split: {rk_human(tuple(chosen_test))} [{test_tag}] [pos={test_pos}, neg={test_neg}, total={test_tot}]")
        except Exception as e:
            warnings.warn(f"keyword-aware split logging failed: {e}")

        return chosen_val, chosen_test

    def _get_or_compute_global_keyword_duration(self, data_path: str, exclude_tasks: list[str]):
        """Compute or retrieve the global max keyword duration across runs.

        Returns (duration_seconds | None, onset_seconds | None, word | None).
        Respects self.min_word_length/self.max_word_length and exclude_tasks for computation,
        but caches a single global value per keyword signature so train/val/test are identical.
        """
        assert self.keyword_set is not None
        signature = self.keyword_signature
        cache = self._load_keyword_cache(data_path)
        cache.setdefault("keywords", {})
        kw_entry = cache["keywords"].setdefault(signature, {})

        durations_entry = kw_entry.setdefault("durations", {})
        # Prefer a single global cached value regardless of exclude_tasks
        if "global" in durations_entry:
            d = durations_entry["global"]
            return d.get("duration"), d.get("onset"), d.get("word"), "cache"
        # Back-compat: if any previous keyed entries exist, consolidate to global
        if len(durations_entry) > 0:
            try:
                # Pick the maximum duration among any stored entries
                candidates = [v for v in durations_entry.values() if isinstance(v, dict) and "duration" in v]
                if candidates:
                    best = max(candidates, key=lambda x: x.get("duration", -1))
                    durations_entry.clear()
                    durations_entry["global"] = {"duration": best.get("duration"), "onset": best.get("onset"), "word": best.get("word")}
                    cache["keywords"][signature] = kw_entry
                    self._save_keyword_cache(data_path, cache)
                    return best.get("duration"), best.get("onset"), best.get("word"), "cache-migrated"
            except Exception:
                pass

        # Build run list respecting exclude_tasks
        runs = [tuple(rk) for rk in RUN_KEYS if rk[2] not in set(exclude_tasks)]

        max_dur = -1.0
        max_onset = None
        max_word = None

        for rk in runs:
            events_path = self._events_path_from_run_key(data_path, rk)
            try:
                if self.download:
                    LibriBrainBase.ensure_file_download(events_path, data_path)
                elif not os.path.exists(events_path):
                    continue
                df = pd.read_csv(events_path, sep="\t")
            except Exception:
                continue

            if "kind" in df.columns:
                df = df[df["kind"] == "word"].copy()
            if "segment" not in df.columns or "duration" not in df.columns or "timemeg" not in df.columns:
                continue

            df["segment"] = df["segment"].astype(str).str.strip()
            if self.min_word_length > 1:
                df = df[df["segment"].str.len() >= self.min_word_length]
            if self.max_word_length is not None:
                df = df[df["segment"].str.len() <= self.max_word_length]

            mask = df["segment"].str.lower().isin(self.keyword_set)
            if not mask.any():
                continue
            df_kw = df.loc[mask].copy()
            df_kw["duration"] = pd.to_numeric(df_kw["duration"], errors="coerce")
            df_kw["timemeg"] = pd.to_numeric(df_kw["timemeg"], errors="coerce")
            df_kw = df_kw.dropna(subset=["duration", "timemeg"])  # require both
            if df_kw.empty:
                continue

            idx_local = df_kw["duration"].idxmax()
            row = df_kw.loc[idx_local]
            dur = float(row["duration"])  # seconds
            if dur > max_dur:
                max_dur = dur
                max_onset = float(row["timemeg"])  # seconds
                max_word = str(row["segment"]) if isinstance(row["segment"], str) else None

        if max_dur >= 0:
            durations_entry["global"] = {"duration": max_dur, "onset": max_onset, "word": max_word}
            cache["keywords"][signature] = kw_entry
            self._save_keyword_cache(data_path, cache)
            return max_dur, max_onset, max_word, "computed"

        return None, None, None, None

    def _get_unique_word_labels(self):
        """Extract unique words from all samples and sort them."""
        words = set()
        for sample in self.samples:
            word = sample[5]  # word is at index 5
            # Be robust to non-string entries (e.g., NaN floats)
            if isinstance(word, str):
                words.add(word)
        words = list(words)
        words.sort()
        return words

    def load_words_from_tsv(self, subject, session, task, run):
        """Load word events from the TSV file."""
        events_df = self._load_events(subject, session, task, run)
        
        # Filter for word events only
        word_events = events_df[events_df["kind"] == "word"].copy()

        # Drop rows with missing word labels/onsets and normalize to strings
        if "segment" in word_events.columns:
            # Ensure we have valid word and onset values
            drop_cols = ["segment"]
            if "timemeg" in word_events.columns:
                drop_cols.append("timemeg")
            word_events = word_events.dropna(subset=drop_cols).copy()
            # Ensure consistent string type and trim whitespace
            word_events["segment"] = word_events["segment"].astype(str).str.strip()
        
        # Apply word length filtering
        if self.min_word_length > 1:
            word_events = word_events[word_events["segment"].str.len() >= self.min_word_length]
        if self.max_word_length is not None:
            word_events = word_events[word_events["segment"].str.len() <= self.max_word_length]
        
        # Extract relevant information
        words = word_events["segment"].values
        onsets = word_events["timemeg"].values
        sentence_indices = word_events["sentenceidx"].values
        word_indices = word_events["wordidx"].values
        
        return words, onsets, sentence_indices, word_indices

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        # Extract word information from the sample
        subject, session, task, run, onset, word, sent_idx, word_idx = self.samples[idx]
        
        # Get base data from parent class using a modified sample
        original_sample = self.samples[idx]
        # Temporarily modify sample to match base class expectation (6 elements)
        self.samples[idx] = (subject, session, task, run, onset, word)
        
        # Get base data from parent class
        result = super().__getitem__(idx)
        data, label, info = result
        
        # Restore original sample
        self.samples[idx] = original_sample
        
        # Add word-specific info
        if self.include_info:
            info.update({
                "word": word,
                "sentence_idx": sent_idx,
                "word_idx": word_idx,
            })

        # Convert word to appropriate label
        if self.keyword_set is not None:
            # Binary classification: keyword vs other (supports multiple keywords)
            label_id = 1 if isinstance(word, str) and word.lower() in self.keyword_set else 0
        else:
            # Multi-class classification: word ID
            label_id = self.word_to_id[word]
        
        if self.include_info:
            return [data, torch.tensor(label_id), info]
        return [data, torch.tensor(label_id)]


if __name__ == "__main__":
    import time

    start_time = time.time()
    val_dataset = LibriBrainWord(
        data_path="/Users/mirgan/LibriBrain/serialized/",
        partition="validation",
        preload_files=False,
    )
    test_dataset = LibriBrainWord(
        data_path="/Users/mirgan/LibriBrain/serialized/",
        partition="test",
        preload_files=False,
    )
    print("len(val_dataset): ", len(val_dataset))
    print("len(test_dataset): ", len(test_dataset))

    print(f"Unique words in validation set: {len(val_dataset.words_sorted)}")
    print(f"First 10 words: {val_dataset.words_sorted[:10]}")

    # Count word frequencies
    word_counts = torch.zeros(len(val_dataset.labels_sorted))
    start_time = time.time()
    for i in range(len(val_dataset)):
        _, word_id = val_dataset[i]
        word_counts[word_id] += 1
        if i % 1000 == 0:
            print(f"Processed {i} samples in {time.time() - start_time:.2f}s")
            start_time = time.time()
    
    print(f"Word frequencies: {word_counts}") 