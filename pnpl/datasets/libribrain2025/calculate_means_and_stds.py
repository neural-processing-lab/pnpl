import numpy as np
from pnpl.datasets import LibriBrainPhoneme
import json
import os
from pnpl.datasets.libribrain2025.base import LibriBrainBase
import torch
from pnpl.datasets.grouped_dataset import GroupedDataset

output_path = "/data/engs-pnpl/trin4076/pnpl/pnpl/datasets/libribrain2025"
data_path = "/data/engs-pnpl/datasets/LibriBrain/serialized/default"
dataset = LibriBrainPhoneme(
    data_path=data_path,
    partition="train",
    standardize=True,
)

h5_standardization_params = {}
h5_standardization_params["channel_means"] = dataset.channel_means.tolist()
h5_standardization_params["channel_stds"] = dataset.channel_stds.tolist()

with open(os.path.join(output_path, "h5_standardization_params.json"), "w") as f:
    json.dump(h5_standardization_params, f)

print("H5 standardization parameters saved to:",
      os.path.join(output_path, "h5_standardization_params.json"))


grouped_dataset = GroupedDataset(
    dataset, grouped_samples=100, average_grouped_samples=True)
means = []
stds = []
for i in range(len(dataset)):
    means.append(dataset[i][0].mean(dim=-1))
    stds.append(dataset[i][0].std(dim=-1))
means = torch.stack(means, dim=0)
stds = torch.stack(stds, dim=0)
stds, means = LibriBrainBase._accumulate_stds(
    means.numpy(), stds.numpy(), (torch.ones(len(means)) * dataset[0][0].shape[1]).numpy())

grouped_standardization_params = {}
grouped_standardization_params["channel_means"] = means.tolist()
grouped_standardization_params["channel_stds"] = stds.tolist()

with open(os.path.join(output_path, "grouped_standardization_params.json"), "w") as f:
    json.dump(grouped_standardization_params, f)

print("Grouped standardization parameters saved to:",
      os.path.join(output_path, "grouped_standardization_params.json"), " Ideally they would be 0, 1")

dataset = LibriBrainPhoneme(
    data_path=data_path,
    partition="train",
    standardize=False,
)

means = []
stds = []
for i in range(len(dataset)):
    means.append(dataset[i][0].mean(dim=-1))
    stds.append(dataset[i][0].std(dim=-1))

means = torch.stack(means, dim=0)
stds = torch.stack(stds, dim=0)
stds, means = LibriBrainBase._accumulate_stds(
    means.numpy(), stds.numpy(), (torch.ones(len(means)) * dataset[0][0].shape[1]).numpy())

ideal_standardization_params = {}
ideal_standardization_params["channel_means"] = means.tolist()
ideal_standardization_params["channel_stds"] = stds.tolist()

with open(os.path.join(output_path, "ideal_standardization_params.json"), "w") as f:
    json.dump(ideal_standardization_params, f)

print("Ideal standardization parameters saved to:",
      os.path.join(output_path, "ideal_standardization_params.json"))

grouped_dataset = GroupedDataset(
    dataset, grouped_samples=100, average_grouped_samples=True)
means = []
stds = []
for i in range(len(grouped_dataset)):
    means.append(grouped_dataset[i][0].mean(dim=-1))
    stds.append(grouped_dataset[i][0].std(dim=-1))

means = torch.stack(means, dim=0)
stds = torch.stack(stds, dim=0)
stds, means = LibriBrainBase._accumulate_stds(
    means.numpy(), stds.numpy(), (torch.ones(len(means)) * grouped_dataset[0][0].shape[1]).numpy())

ideal_grouped_standardization_params = {}
ideal_grouped_standardization_params["channel_means"] = means.tolist()
ideal_grouped_standardization_params["channel_stds"] = stds.tolist()

with open(os.path.join(output_path, "ideal_grouped_standardization_params.json"), "w") as f:
    json.dump(ideal_grouped_standardization_params, f)

# sbatch --account=engs-tvg --qos=tvg --mem=256G scripts/cpu_short_submit.sh pnpl/datasets/libribrain2025/calculate_means_and_stds.py


ideal_means = np.array(ideal_standardization_params["channel_means"])
ideal_stds = np.array(ideal_standardization_params["channel_stds"])

h5_means = np.array(h5_standardization_params["channel_means"])
h5_stds = np.array(h5_standardization_params["channel_stds"])

print("ideal_means: ", ideal_means)
print("h5_means: ", h5_means)


print("ideal_stds: ", ideal_stds)
print("h5_stds: ", h5_stds)

print("means diff mean: ", np.mean(h5_means - ideal_means))
print("means diff max: ", np.max(h5_means - ideal_means))


print("stds diff mean: ", np.mean(h5_stds - ideal_stds))
print("stds diff max: ", np.max(h5_stds - ideal_stds))


std_factor = ideal_stds / h5_stds
print("std_factor: ", std_factor)

print("std_factor min: ", np.min(std_factor))
print("std_factor max: ", np.max(std_factor))
print("std_factor mean: ", np.mean(std_factor))
