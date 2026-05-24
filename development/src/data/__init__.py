"""Data loading utilities."""

import os

from torch.utils.data import DataLoader

from .cifar10 import get_cifar10_loaders


def _resolve_num_workers(num_workers: int) -> int:
    override = os.getenv("DATALOADER_NUM_WORKERS")
    if override is None:
        return num_workers
    try:
        resolved = int(override)
    except ValueError as exc:
        raise ValueError("DATALOADER_NUM_WORKERS must be an integer") from exc
    if resolved < 0:
        raise ValueError("DATALOADER_NUM_WORKERS must be >= 0")
    return resolved


def get_data_loaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: str = "./data",
    val_size: int = 5000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return CIFAR-10 train, validation, and test loaders."""
    num_workers = _resolve_num_workers(num_workers)

    if dataset == "cifar10":
        return get_cifar10_loaders(
            batch_size=batch_size, num_workers=num_workers,
            data_root=data_root, val_size=val_size, seed=seed,
        )
    raise ValueError(f"Unknown dataset: {dataset!r}. This repository supports 'cifar10'.")
