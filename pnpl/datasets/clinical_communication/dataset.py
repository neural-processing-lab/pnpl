"""Clinical communication sentences with the original frozen MEG donors."""
from collections import OrderedDict
from copy import deepcopy
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import asset_path, load, normalize_partition
from .recordings import RecordingStore


class ClinicalCommunication(Dataset):
    """Load the exact single-subject LibriBrain clinical benchmark.

    Train/val default to individual words from the original recording split.
    Use unit="sentence" for natural sentence windows, or unit="group" for
    the original disjoint five-occurrence training/validation groups.
    Test/dev always return constructed clinical sentences.

    k selects the first k of five frozen donors, without replacement,
    resampling, averaging, or reordering. It applies to test/dev and groups;
    natural words/sentences have one occurrence per word. Test defaults
    to the original 100 sentences; test_sentences=200 adds the extension.
    partition="dev" provides 50 development sentences from val recordings.

    Every item is a dict with meg, words, sentence, sentence_id, occurrences,
    and positions. MEG shapes:
      word: (306, 150); group: (k, 306, 150);
      natural sentence: (words, 306, 150);
      clinical sentence: (words, k, 306, 150).
    Words are strings, with no model-specific target embeddings. Variable-length
    sentences need a custom collate function (or batch_size=None).
    metadata(index) inspects exact donor identities without recording I/O.

    Downloads use pinned revisions and preprocessing is cached per recording.
    download=False uses local sources/cache only. Construction never downloads.
    """

    def __init__(self, data_path, partition="train", *, k=5, test_sentences=100,
                 unit="word", cache_path=None, download=True):
        self.partition = normalize_partition(partition)
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 5:
            raise ValueError("k must be an integer from 1 to 5")
        if isinstance(test_sentences, bool) or test_sentences not in (100, 200):
            raise ValueError("test_sentences must be 100 or 200")
        if unit not in {"word", "sentence", "group"}:
            raise ValueError("unit must be word, sentence, or group")
        if self.partition in {"test", "dev"} and unit == "group":
            raise ValueError("test/dev return clinical sentences; unit=group is train/val only")
        self.k = int(k)
        self.test_sentences = test_sentences
        self.unit = "clinical" if self.partition in {"test", "dev"} else unit
        self.source_split = "val" if self.partition == "dev" else self.partition
        self._natural = load(f"natural_{self.source_split}.json.gz")
        self._records = load("recordings.json")
        self._groups = None
        if self.unit in {"clinical", "group"}:
            self._groups = load("groups.json.gz")[self.partition]
        if self.unit == "clinical":
            grouped = OrderedDict()
            for group in self._groups:
                if self.partition == "test" and group["sentence_id"] > test_sentences:
                    continue
                grouped.setdefault(group["sentence_id"], []).append(group)
            self._sentences = list(grouped.items())
            if self.partition == "dev":
                self._texts = {i: text for i, text in enumerate(load("development_sentences.json"), 1)}
            else:
                self._texts = {int(r["sentence_id"]): r["sentence"] for r in load("sentences.json")}
        self.store = RecordingStore(data_path, cache_path=cache_path, download=download)
        self.positions = torch.from_numpy(np.load(asset_path("positions.npy")).copy())
        self.candidate_words = sorted({g["word"] for g in load("groups.json.gz")["test"]})
        self.recordings = deepcopy([r for r in self._records if r["split"] == self.source_split])

    def __len__(self):
        if self.unit == "clinical":
            return len(self._sentences)
        if self.unit == "group":
            return len(self._groups)
        if self.unit == "sentence":
            return len(self._natural["sequences"])
        return len(self._natural["items"])

    def _selection(self, index):
        if not isinstance(index, Integral):
            raise TypeError("dataset indices must be integers")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.unit == "clinical":
            sid, groups = self._sentences[index]
            return [g["indices"][:self.k] for g in groups], sid, self._texts[sid]
        if self.unit == "group":
            return [self._groups[index]["indices"][:self.k]], None, self._groups[index]["word"]
        if self.unit == "sentence":
            indices = self._natural["sequences"][index]
            return [[i] for i in indices], index, " ".join(self._natural["items"][i]["word"] for i in indices)
        item = self._natural["items"][index]
        return [[index]], None, item["word"]

    def metadata(self, index):
        """Return portable source paths, event row IDs and exact donor order."""
        selected, sid, text = self._selection(index)
        words, occurrences = [], []
        for indices in selected:
            items = [self._natural["items"][i] for i in indices]
            word = items[0]["word"]
            if any(i["word"] != word for i in items):
                raise ValueError("Frozen occurrence word mismatch")
            words.append(word)
            donors = []
            for member, item in enumerate(items):
                record = self._records[item["record"]]
                donors.append(dict(record=item["record"], neural=record["neural"],
                    events=record["events"], event_index=item["event"],
                    onset=item["onset"], published_onset=item["onset"] + record["origin"],
                    duration=item["duration"], subject=record["subject"],
                    session=record["session"], member=member))
            occurrences.append(donors)
        return dict(words=words, sentence=text, sentence_id=sid,
                    occurrences=occurrences, partition=self.partition,
                    source_split=self.source_split)

    def __getitem__(self, index):
        result = self.metadata(index)
        windows = [torch.stack([self.store.window(
            self._records[d["record"]], d["onset"]) for d in donors])
            for donors in result["occurrences"]]
        meg = torch.stack(windows)
        if self.unit == "word":
            meg = meg[0, 0]
        elif self.unit == "group":
            meg = meg[0]
        elif self.unit == "sentence":
            meg = meg[:, 0]
        result.update(meg=meg, positions=self.positions.clone())
        return result

    def prepare(self):
        """Download/preprocess only recordings used by this dataset view."""
        if self.unit == "clinical":
            indices = {i for _, groups in self._sentences
                       for g in groups for i in g["indices"][:self.k]}
        elif self.unit == "group":
            indices = {i for g in self._groups for i in g["indices"][:self.k]}
        else:
            indices = range(len(self._natural["items"]))
        record_ids = {self._natural["items"][i]["record"] for i in indices}
        for rid in sorted(record_ids):
            self.store.array(self._records[rid])

    def close(self):
        self.store.close()
