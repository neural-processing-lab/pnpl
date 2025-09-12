from torch.utils.data import Dataset, DataLoader
import time


class InMemoryDataset(Dataset):
    def __init__(self, original_dataset):
        """
        Initializes the InMemoryDataset by loading all data into memory in a single batch.

        Parameters:
        - dataset_class: A PyTorch Dataset class (e.g., torchvision.datasets.ImageFolder)
        - dataset_config: A dictionary of configuration parameters to initialize the dataset_class
        """
        # Instantiate the original dataset
        dataset_length = len(original_dataset)

        start_time = time.time()
        # Create a DataLoader to load the entire dataset in one batch
        self.data = []
        for i in range(dataset_length):
            self.data.append(original_dataset[i])
        end_time = time.time()
        print(f"Preloading completed. Total samples loaded: {len(self.data)}")
        print(f"Time taken to preload: {end_time - start_time:.2f} seconds")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
