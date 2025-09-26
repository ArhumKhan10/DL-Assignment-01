"""Create train/validation/test splits for the affect dataset."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    root_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_by_expression: bool = True,
) -> Dict[str, List[str]]:
    """Create train/val/test splits for the dataset.
    
    Args:
        root_dir: Root directory containing Dataset/annotations
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation  
        test_ratio: Fraction of data for testing
        random_seed: Random seed for reproducibility
        stratify_by_expression: Whether to stratify by expression class
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing lists of sample IDs
    """
    root = Path(root_dir)
    ann_dir = root / "Dataset" / "annotations"
    
    # Get all available sample IDs
    all_ids = sorted([p.name.replace("_exp.npy", "") for p in ann_dir.glob("*_exp.npy")])
    
    print(f"Found {len(all_ids)} total samples")
    
    # Load expression labels for stratification
    expressions = []
    valid_ids = []
    
    for sid in all_ids:
        exp_path = ann_dir / f"{sid}_exp.npy"
        val_path = ann_dir / f"{sid}_val.npy"
        aro_path = ann_dir / f"{sid}_aro.npy"
        
        if not (exp_path.exists() and val_path.exists() and aro_path.exists()):
            continue
            
        try:
            exp = int(np.load(exp_path))
            val = float(np.load(val_path))
            aro = float(np.load(aro_path))
            
            # Skip uncertain samples (val == -2 or aro == -2)
            if val == -2 or aro == -2:
                continue
                
            expressions.append(exp)
            valid_ids.append(sid)
        except Exception:
            continue
    
    print(f"Valid samples (excluding uncertain): {len(valid_ids)}")
    
    # Create splits
    if stratify_by_expression and len(set(expressions)) > 1:
        # Stratified split
        train_ids, temp_ids, train_expr, temp_expr = train_test_split(
            valid_ids, expressions, 
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            stratify=expressions
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_expr,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_seed,
            stratify=temp_expr
        )
    else:
        # Random split
        train_ids, temp_ids = train_test_split(
            valid_ids, 
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_seed
        )
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Print class distribution
    for split_name, ids in splits.items():
        split_expr = []
        for sid in ids:
            idx = valid_ids.index(sid)
            split_expr.append(expressions[idx])
        
        unique, counts = np.unique(split_expr, return_counts=True)
        print(f"{split_name} expression distribution:")
        for expr, count in zip(unique, counts):
            print(f"  Class {expr}: {count} samples")
    
    return splits


def save_splits(splits: Dict[str, List[str]], output_path: str | Path) -> None:
    """Save splits to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Splits saved to {output_path}")


def load_splits(splits_path: str | Path) -> Dict[str, List[str]]:
    """Load splits from JSON file."""
    with open(splits_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Create splits for the current dataset
    root_dir = Path(__file__).resolve().parents[2]
    splits = create_splits(root_dir)
    
    # Save splits
    output_path = root_dir / "data" / "splits.json"
    save_splits(splits, output_path)