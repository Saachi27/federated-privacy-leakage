"""Dataset loading and client partitioning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-dataset metadata so the rest of the code stays dataset-agnostic.
DATASET_META = {
    "mnist": {"in_channels": 1, "num_classes": 10, "input_size": 28},
    "cifar10": {"in_channels": 3, "num_classes": 10, "input_size": 32},
}


def load_datasets(name, data_root="./data", download=True, normalize=True):
    """Return (train_dataset, test_dataset, meta) for a named dataset."""
    name = name.lower()
    if name == "mnist":
        steps = [transforms.ToTensor()]
        if normalize:
            steps.append(transforms.Normalize((0.1307,), (0.3081,)))
        tf = transforms.Compose(steps)
        train = datasets.MNIST(data_root, train=True, download=download, transform=tf)
        test = datasets.MNIST(data_root, train=False, download=download, transform=tf)
    elif name == "cifar10":
        steps = [transforms.ToTensor()]
        if normalize:
            steps.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)))
        tf = transforms.Compose(steps)
        train = datasets.CIFAR10(data_root, train=True, download=download, transform=tf)
        test = datasets.CIFAR10(data_root, train=False, download=download, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, DATASET_META[name]


def get_sample(name, index=0, data_root="./data", train=False):
    """Fetch one (image, label, meta) in [0, 1] space for the attack demo."""
    train_ds, test_ds, meta = load_datasets(name, data_root, download=True, normalize=False)
    ds = train_ds if train else test_ds
    img, label = ds[index]
    return img, int(label), meta


def partition_iid(num_samples, num_clients, seed=0):
    """Split indices [0, num_samples) into `num_clients` ~equal IID shards."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    return [list(shard) for shard in np.array_split(perm, num_clients)]


def make_client_loaders(train_dataset, client_indices, batch_size):
    """One DataLoader per client over its index shard."""
    loaders = []
    for idx in client_indices:
        subset = Subset(train_dataset, idx)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    return loaders


def make_test_loader(test_dataset, batch_size=256):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
