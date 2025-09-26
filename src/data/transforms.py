"""Data augmentation and preprocessing transforms for the affect dataset."""

from __future__ import annotations

import torch
from torchvision import transforms
from typing import Optional, Tuple


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Get training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Get validation/test transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_pil_transforms() -> transforms.Compose:
    """Get transforms that keep PIL images (for visualization)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


class ToTensor:
    """Custom transform to convert numpy arrays to tensors."""
    
    def __call__(self, sample: dict) -> dict:
        """Convert numpy arrays in sample to tensors."""
        if 'landmarks' in sample:
            sample['landmarks'] = torch.from_numpy(sample['landmarks']).float()
        if 'expression' in sample:
            sample['expression'] = torch.tensor(sample['expression'], dtype=torch.long)
        if 'valence' in sample:
            sample['valence'] = torch.tensor(sample['valence'], dtype=torch.float32)
        if 'arousal' in sample:
            sample['arousal'] = torch.tensor(sample['arousal'], dtype=torch.float32)
        return sample

