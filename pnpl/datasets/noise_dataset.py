from torch.utils.data import Dataset

import torch


class NoiseDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        result = self.original_dataset[idx]
        result[0] = torch.randn_like(result[0])
        return result
