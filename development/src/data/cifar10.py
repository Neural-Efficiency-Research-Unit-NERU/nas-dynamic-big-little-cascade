import random

import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
    val_size: int = 5000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, test_transform = get_transforms()

    full_train = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    val_set = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=test_transform
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # Split: 45K train / 5K val
    indices = list(range(len(full_train)))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_set, val_indices)
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader, test_loader
