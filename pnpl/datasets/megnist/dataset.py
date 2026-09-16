"""
MegNIST dataset.

MegNIST contains MEG recordings collected while a participant imagined
speaking the digit names zero through nine. The public dataset is hosted
at ``pnpl/MegNIST`` on Hugging Face.

This loader provides PyTorch-compatible access to the released
``derivatives/serialised/{train,val,test}.h5`` splits. Raw BIDS data and
intermediate preprocessing derivatives are also available in the dataset
repository.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..mixins import EpochedH5Mixin, HFDownloadMixin, StandardizationMixin
from ...tasks.megnist import DigitClassification


class MegNIST(
    HFDownloadMixin,
    StandardizationMixin,
    EpochedH5Mixin,
    Dataset,
):
    """MegNIST imagined-digit MEG dataset.

    Args:
        data_path: Path to store/load the dataset.
        task: Task object defining sample collection and labelling.
            Defaults to :class:`pnpl.tasks.megnist.DigitClassification`.
        partition: ``"train"``, ``"validation"`` (or ``"val"``), or
            ``"test"``.
        preprocessing: Preprocessing identifier. ``None`` selects the
            released canonical HDF5 splits.
        preprocessing_config: Optional preprocessing-step configuration.
            Reserved for generation of alternative derivatives from the
            released raw/intermediate data.
        standardize: Whether to apply PNPL channel-wise standardisation.
            Defaults to False for MegNIST.
        clipping_boundary: Clip standardised values to
            ``[-clipping_boundary, clipping_boundary]``.
        channel_means: Optional pre-computed channel means.
        channel_stds: Optional pre-computed channel standard deviations.
        include_info: If True, return ``(data, label, info)``.
        download: If True, download missing released files from Hugging Face.
    """

    HUGGINGFACE_REPO = "pnpl/MegNIST"

    def __init__(
        self,
        data_path: str,
        task: Optional[Any] = None,
        partition: str = "train",
        preprocessing: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
        standardize: bool = False,
        clipping_boundary: Optional[float] = 10.0,
        channel_means: Optional[np.ndarray] = None,
        channel_stds: Optional[np.ndarray] = None,
        include_info: bool = False,
        download: bool = True,
    ):
        os.makedirs(data_path, exist_ok=True)

        if partition == "validation":
            partition = "val"
        if partition not in ("train", "val", "test"):
            raise ValueError(
                "partition must be 'train', 'validation'/'val', or 'test'"
            )

        self.data_path = data_path
        self.partition = partition
        self.preprocessing = preprocessing
        self.preprocessing_config = preprocessing_config or {}
        self.include_info = include_info
        self.download = download
        self.task = task if task is not None else DigitClassification()

        self.init_epoched_data(partition, preprocessing)

        self.samples = self.task.collect_samples(self)
        if not self.samples:
            raise ValueError(
                f"No MegNIST samples found for partition={partition!r}"
            )

        if self.times is not None and len(self.times) > 1:
            self.sfreq = float(round(1.0 / np.median(np.diff(self.times))))
        else:
            self.sfreq = 250.0

        self.points_per_sample = self.n_times

        self.setup_standardization(
            standardize=standardize,
            clipping_boundary=clipping_boundary,
            channel_means=channel_means,
            channel_stds=channel_stds,
        )

        if standardize and channel_means is None and channel_stds is None:
            self._calculate_standardization_params()

    def _calculate_standardization_params(self) -> None:
        """Calculate channel-wise mean and standard deviation."""
        data = self._epoched_data  # trials x channels x time
        n_channels = data.shape[1]
        reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)

        self.channel_means = np.mean(reshaped, axis=1)
        self.channel_stds = np.std(reshaped, axis=1)

        self.channel_stds[self.channel_stds == 0] = 1.0

        self.broadcasted_means = np.tile(
            self.channel_means, (self.n_times, 1)
        ).T
        self.broadcasted_stds = np.tile(
            self.channel_stds, (self.n_times, 1)
        ).T

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self.samples)}"
            )

        sample = self.samples[idx]
        trial_idx = int(sample[0])

        data = self.get_epoch(trial_idx)
        data = self.standardize(data)

        label = self.task.get_label(sample)

        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label)

        if self.include_info:
            info = {
                "dataset": "megnist",
                "partition": self.partition,
                "trial_idx": trial_idx,
            }
            return data, label, info

        return data, label

    @property
    def label_info(self):
        return self.task.label_info