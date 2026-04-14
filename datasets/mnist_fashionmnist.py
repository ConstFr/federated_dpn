import math
import logging

import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def make_mnist_fashionmnist_datasets(data_path="./data", normalize=True, train_subset_size=None):
    if normalize:
        mean = (0.1307,)
        std = (0.3081,)
    else:
        mean = (0.5,)
        std = (0.5,)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ood_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.MNIST(root=data_path, train=True, transform=train_transform, download=True)
    val_dataset = datasets.MNIST(root=data_path, train=False, transform=eval_transform, download=True)
    ood_train_dataset = datasets.FashionMNIST(root=data_path, train=True, transform=ood_transform, download=True)
    ood_val_dataset = datasets.FashionMNIST(root=data_path, train=False, transform=eval_transform, download=True)

    if train_subset_size is not None:
        indices = np.random.permutation(len(train_dataset))[:train_subset_size]
        train_dataset = data.Subset(train_dataset, indices)
        ood_train_dataset = data.Subset(ood_train_dataset, indices)

    if len(val_dataset) != len(ood_val_dataset):
        min_val_len = min(len(val_dataset), len(ood_val_dataset))
        val_dataset = data.Subset(val_dataset, np.arange(min_val_len))
        ood_val_dataset = data.Subset(ood_val_dataset, np.arange(min_val_len))

    n_id = len(train_dataset)
    n_ood = len(ood_train_dataset)
    if n_id < n_ood:
        repeat = math.ceil(n_ood / n_id)
        train_dataset = data.ConcatDataset([train_dataset] * repeat)
        train_dataset = data.Subset(train_dataset, range(n_ood))
    elif n_ood < n_id:
        repeat = math.ceil(n_id / n_ood)
        ood_train_dataset = data.ConcatDataset([ood_train_dataset] * repeat)
        ood_train_dataset = data.Subset(ood_train_dataset, range(n_id))

    assert len(train_dataset) == len(ood_train_dataset)
    if len(train_dataset) != len(ood_train_dataset):
        raise RuntimeError(
            f"Balanced train dataset sizes still differ: {len(train_dataset)} != {len(ood_train_dataset)}"
        )

    logger.info(f"Validation dataset length: {len(val_dataset)}")
    logger.info(f"OOD validation dataset length: {len(ood_val_dataset)}")
    logger.info(f"Train dataset length: {len(train_dataset)}")
    logger.info(f"OOD train dataset length: {len(ood_train_dataset)}")

    return train_dataset, val_dataset, ood_train_dataset, ood_val_dataset
