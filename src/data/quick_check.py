"""Quick check script to test data loading pipeline."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import AffectDataset
from data.transforms import get_train_transforms, get_val_transforms, ToTensor


def test_dataset():
    """Test the dataset loading."""
    root_dir = Path(__file__).resolve().parents[2]
    
    print("Testing AffectDataset...")
    
    # Test without transforms
    dataset = AffectDataset(root_dir, return_pil=True)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Image type: {type(sample['image'])}")
        print(f"Expression: {sample['expression']}")
        print(f"Valence: {sample['valence']}")
        print(f"Arousal: {sample['arousal']}")
        print(f"Landmarks shape: {sample['s'].shape}")
        print(f"Sample ID: {sample['id']}")
    
    # Test with transforms
    print("\nTesting with transforms...")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    to_tensor = ToTensor()
    
    train_dataset = AffectDataset(root_dir, transform=train_transform)
    val_dataset = AffectDataset(root_dir, transform=val_transform)
    
    if len(train_dataset) > 0:
        train_sample = train_dataset[0]
        print(f"Train sample image shape: {train_sample['image'].shape}")
        print(f"Train sample image dtype: {train_sample['image'].dtype}")
    
    if len(val_dataset) > 0:
        val_sample = val_dataset[0]
        print(f"Val sample image shape: {val_sample['image'].shape}")
        print(f"Val sample image dtype: {val_sample['image'].dtype}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    if len(dataset) > 0:
        # Use a dataset with tensor transforms for batching
        tensor_dataset = AffectDataset(root_dir, transform=val_transform)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for Windows compatibility
            collate_fn=lambda batch: {
                'images': torch.stack([sample['image'] for sample in batch]),
                'expressions': torch.tensor([sample['expression'] for sample in batch], dtype=torch.long),
                'valence': torch.tensor([sample['valence'] for sample in batch], dtype=torch.float32),
                'arousal': torch.tensor([sample['arousal'] for sample in batch], dtype=torch.float32),
                'landmarks': torch.stack([to_tensor(sample)['landmarks'] for sample in batch]),
                'ids': [sample['id'] for sample in batch]
            }
        )
        
        batch = next(iter(dataloader))
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Batch expressions shape: {batch['expressions'].shape}")
        print(f"Batch valence shape: {batch['valence'].shape}")
        print(f"Batch arousal shape: {batch['arousal'].shape}")
        print(f"Batch landmarks shape: {batch['landmarks'].shape}")
        print(f"Batch IDs: {batch['ids']}")


if __name__ == "__main__":
    test_dataset()