"""Compare trained models and generate qualitative results."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.loaders import get_dataloaders
from models.baseline_models import create_resnet_model, create_efficientnet_model
from utils.metrics import compute_all_metrics, print_metrics


def load_metrics(model_dir: Path) -> pd.DataFrame:
    """Load metrics from CSV file."""
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    return df


def get_best_epoch_metrics(df: pd.DataFrame) -> dict:
    """Get metrics from the epoch with best validation loss."""
    best_epoch_idx = df['val_total'].idxmin()
    best_epoch = df.iloc[best_epoch_idx]
    
    return {
        'epoch': int(best_epoch['epoch']),
        'val_loss': float(best_epoch['val_total']),
        'expr_accuracy': float(best_epoch['expr_acc']),
        'expr_f1_macro': float(best_epoch['expr_f1_macro']),
        'val_rmse': float(best_epoch['val_rmse']),
        'val_corr': float(best_epoch['val_corr']),
        'val_sagr': float(best_epoch['val_sagr']),
        'val_ccc': float(best_epoch['val_ccc']),
        'aro_rmse': float(best_epoch['aro_rmse']),
        'aro_corr': float(best_epoch['aro_corr']),
        'aro_sagr': float(best_epoch['aro_sagr']),
        'aro_ccc': float(best_epoch['aro_ccc']),
    }


def plot_training_curves(resnet_df: pd.DataFrame, efficientnet_df: pd.DataFrame, output_dir: Path):
    """Plot training curves for both models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(resnet_df['epoch'], resnet_df['train_total'], label='ResNet50 Train', alpha=0.7)
    axes[0, 0].plot(resnet_df['epoch'], resnet_df['val_total'], label='ResNet50 Val', alpha=0.7)
    axes[0, 0].plot(efficientnet_df['epoch'], efficientnet_df['train_total'], label='EfficientNet Train', alpha=0.7)
    axes[0, 0].plot(efficientnet_df['epoch'], efficientnet_df['val_total'], label='EfficientNet Val', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Expression accuracy
    axes[0, 1].plot(resnet_df['epoch'], resnet_df['expr_acc'], label='ResNet50', alpha=0.7)
    axes[0, 1].plot(efficientnet_df['epoch'], efficientnet_df['expr_acc'], label='EfficientNet', alpha=0.7)
    axes[0, 1].set_title('Expression Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Valence correlation
    axes[0, 2].plot(resnet_df['epoch'], resnet_df['val_corr'], label='ResNet50', alpha=0.7)
    axes[0, 2].plot(efficientnet_df['epoch'], efficientnet_df['val_corr'], label='EfficientNet', alpha=0.7)
    axes[0, 2].set_title('Valence Correlation')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Correlation')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Arousal correlation
    axes[1, 0].plot(resnet_df['epoch'], resnet_df['aro_corr'], label='ResNet50', alpha=0.7)
    axes[1, 0].plot(efficientnet_df['epoch'], efficientnet_df['aro_corr'], label='EfficientNet', alpha=0.7)
    axes[1, 0].set_title('Arousal Correlation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Valence SAGR
    axes[1, 1].plot(resnet_df['epoch'], resnet_df['val_sagr'], label='ResNet50', alpha=0.7)
    axes[1, 1].plot(efficientnet_df['epoch'], efficientnet_df['val_sagr'], label='EfficientNet', alpha=0.7)
    axes[1, 1].set_title('Valence SAGR')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('SAGR')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Arousal SAGR
    axes[1, 2].plot(resnet_df['epoch'], resnet_df['aro_sagr'], label='ResNet50', alpha=0.7)
    axes[1, 2].plot(efficientnet_df['epoch'], efficientnet_df['aro_sagr'], label='EfficientNet', alpha=0.7)
    axes[1, 2].set_title('Arousal SAGR')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('SAGR')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_qualitative_results(model, test_loader, output_dir: Path, model_name: str, num_samples: int = 8):
    """Generate qualitative results showing correct/incorrect predictions."""
    model.eval()
    
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images']
            expressions = batch['expressions']
            valence = batch['valence']
            arousal = batch['arousal']
            ids = batch['ids']
            
            outputs = model(images)
            pred_expressions = outputs['expression'].argmax(dim=1)
            pred_valence = outputs['valence']
            pred_arousal = outputs['arousal']
            
            for i in range(len(images)):
                is_correct = pred_expressions[i].item() == expressions[i].item()
                sample = {
                    'id': ids[i],
                    'image': images[i],
                    'true_expr': expressions[i].item(),
                    'pred_expr': pred_expressions[i].item(),
                    'true_val': valence[i].item(),
                    'pred_val': pred_valence[i].item(),
                    'true_aro': arousal[i].item(),
                    'pred_aro': pred_arousal[i].item(),
                    'correct': is_correct
                }
                
                if is_correct and len(correct_samples) < num_samples:
                    correct_samples.append(sample)
                elif not is_correct and len(incorrect_samples) < num_samples:
                    incorrect_samples.append(sample)
                
                if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                    break
            
            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break
    
    # Plot qualitative results
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    # Correct predictions
    for i, sample in enumerate(correct_samples[:8]):
        ax = axes[i]
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Correct\nID: {sample['id']}\nTrue: {sample['true_expr']}, Pred: {sample['pred_expr']}\nVal: {sample['true_val']:.2f}→{sample['pred_val']:.2f}\nAro: {sample['true_aro']:.2f}→{sample['pred_aro']:.2f}")
        ax.axis('off')
    
    # Incorrect predictions
    for i, sample in enumerate(incorrect_samples[:8]):
        ax = axes[i + 8]
        img = sample['image'].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Incorrect\nID: {sample['id']}\nTrue: {sample['true_expr']}, Pred: {sample['pred_expr']}\nVal: {sample['true_val']:.2f}→{sample['pred_val']:.2f}\nAro: {sample['true_aro']:.2f}→{sample['pred_aro']:.2f}")
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - Qualitative Results', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name.lower()}_qualitative.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison function."""
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    resnet_dir = root / "outputs" / "resnet50_e34"
    efficientnet_dir = root / "outputs" / "efficientnet_e34"
    
    print("Loading metrics...")
    resnet_df = load_metrics(resnet_dir)
    efficientnet_df = load_metrics(efficientnet_dir)
    
    # Get best results
    resnet_best = get_best_epoch_metrics(resnet_df)
    efficientnet_best = get_best_epoch_metrics(efficientnet_df)
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Val Loss', 'Expr Accuracy', 'Expr F1-Macro', 'Val RMSE', 'Val Corr', 'Val SAGR', 'Val CCC', 'Aro RMSE', 'Aro Corr', 'Aro SAGR', 'Aro CCC'],
        'ResNet50': [
            f"{resnet_best['val_loss']:.4f}",
            f"{resnet_best['expr_accuracy']:.4f}",
            f"{resnet_best['expr_f1_macro']:.4f}",
            f"{resnet_best['val_rmse']:.4f}",
            f"{resnet_best['val_corr']:.4f}",
            f"{resnet_best['val_sagr']:.4f}",
            f"{resnet_best['val_ccc']:.4f}",
            f"{resnet_best['aro_rmse']:.4f}",
            f"{resnet_best['aro_corr']:.4f}",
            f"{resnet_best['aro_sagr']:.4f}",
            f"{resnet_best['aro_ccc']:.4f}",
        ],
        'EfficientNet-B0': [
            f"{efficientnet_best['val_loss']:.4f}",
            f"{efficientnet_best['expr_accuracy']:.4f}",
            f"{efficientnet_best['expr_f1_macro']:.4f}",
            f"{efficientnet_best['val_rmse']:.4f}",
            f"{efficientnet_best['val_corr']:.4f}",
            f"{efficientnet_best['val_sagr']:.4f}",
            f"{efficientnet_best['val_ccc']:.4f}",
            f"{efficientnet_best['aro_rmse']:.4f}",
            f"{efficientnet_best['aro_corr']:.4f}",
            f"{efficientnet_best['aro_sagr']:.4f}",
            f"{efficientnet_best['aro_ccc']:.4f}",
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    print("\nModel Comparison (Best Epoch Results):")
    print(comparison_df.to_string(index=False))
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(resnet_df, efficientnet_df, output_dir)
    
    # Generate qualitative results
    print("Generating qualitative results...")
    
    # Load test data
    splits_path = root / "data" / "splits.json"
    _, _, test_loader = get_dataloaders(
        root_dir=root,
        splits_path=splits_path,
        batch_size=32,
        num_workers=0,  # Use 0 for qualitative analysis
    )
    
    # Load best models
    resnet_model = create_resnet_model(pretrained=False)
    resnet_checkpoint = torch.load(resnet_dir / 'best_model.pth', map_location='cpu')
    resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
    
    efficientnet_model = create_efficientnet_model(pretrained=False)
    efficientnet_checkpoint = torch.load(efficientnet_dir / 'best_model.pth', map_location='cpu')
    efficientnet_model.load_state_dict(efficientnet_checkpoint['model_state_dict'])
    
    # Generate qualitative results
    generate_qualitative_results(resnet_model, test_loader, output_dir, "ResNet50")
    generate_qualitative_results(efficientnet_model, test_loader, output_dir, "EfficientNet")
    
    print(f"\nResults saved to: {output_dir}")
    print("- model_comparison.csv: Quantitative comparison")
    print("- training_curves.png: Training progress plots")
    print("- resnet50_qualitative.png: ResNet50 sample predictions")
    print("- efficientnet_qualitative.png: EfficientNet sample predictions")


if __name__ == "__main__":
    main()

