"""Reusable data loader utilities for the affect dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import AffectDataset
from .transforms import get_train_transforms, get_val_transforms, ToTensor
from .make_splits import load_splits


def collate_samples(batch: list[Dict[str, object]]) -> Dict[str, torch.Tensor | list[str]]:
    """Top-level collate function (picklable for multiprocessing on Windows)."""
    to_tensor = ToTensor()
    return {
        "images": torch.stack([sample["image"] for sample in batch]),
        "expressions": torch.tensor([sample["expression"] for sample in batch], dtype=torch.long),
        "valence": torch.tensor([sample["valence"] for sample in batch], dtype=torch.float32),
        "arousal": torch.tensor([sample["arousal"] for sample in batch], dtype=torch.float32),
        "landmarks": torch.stack([to_tensor(sample)["landmarks"] for sample in batch]),
        "ids": [sample["id"] for sample in batch],
    }


def get_dataloaders(
    root_dir: str | Path,
    splits_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders using saved splits.

    Args:
        root_dir: Project root (contains `Dataset/`)
        splits_path: Path to JSON splits file
        batch_size: Batch size for all loaders
        num_workers: DataLoader workers (0 on Windows)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    root = Path(root_dir)
    splits = load_splits(splits_path)

    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_ds = AffectDataset(root, transform=train_transform, ids=splits["train"]) 
    val_ds = AffectDataset(root, transform=val_transform, ids=splits["val"]) 
    test_ds = AffectDataset(root, transform=val_transform, ids=splits["test"]) 

    collate_fn = collate_samples

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


