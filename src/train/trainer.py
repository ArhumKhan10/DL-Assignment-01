"""Training loop and utilities for affect recognition models."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..utils.metrics import compute_all_metrics, print_metrics


class AffectTrainer:
    """Trainer class for affect recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        expression_weight: float = 1.0,
        valence_weight: float = 1.0,
        arousal_weight: float = 1.0,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        # Label smoothing to improve calibration/robustness
        self.expression_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.valence_criterion = nn.MSELoss()
        self.arousal_criterion = nn.MSELoss()
        
        # Optimizer
        # AdamW typically generalizes better than Adam
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Cosine LR decay with warmup-like effect via small T_0
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
        # Loss weights
        self.expression_weight = expression_weight
        self.valence_weight = valence_weight
        self.arousal_weight = arousal_weight
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # Optional limits for fast sanity runs
        self.max_train_batches = max_train_batches
        self.max_val_batches = max_val_batches

        # AMP scaler
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and (self.device == 'cuda'))
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute total loss and individual losses."""
        # Expression loss
        expr_loss = self.expression_criterion(outputs['expression'], targets['expression'])
        
        # Valence loss
        val_loss = self.valence_criterion(outputs['valence'], targets['valence'])
        
        # Arousal loss
        aro_loss = self.arousal_criterion(outputs['arousal'], targets['arousal'])
        
        # Total weighted loss
        total_loss = (
            self.expression_weight * expr_loss +
            self.valence_weight * val_loss +
            self.arousal_weight * aro_loss
        )
        
        return {
            'total': total_loss,
            'expression': expr_loss,
            'valence': val_loss,
            'arousal': aro_loss
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'total': 0, 'expression': 0, 'valence': 0, 'arousal': 0}
        all_predictions = {'expression': [], 'valence': [], 'arousal': []}
        all_targets = {'expression': [], 'valence': [], 'arousal': []}
        
        pbar = tqdm(self.train_loader, desc='Training')
        for step, batch in enumerate(pbar, start=1):
            # Move to device
            images = batch['images'].to(self.device)
            targets = {
                'expression': batch['expressions'].to(self.device),
                'valence': batch['valence'].to(self.device),
                'arousal': batch['arousal'].to(self.device)
            }
            # Step scheduler per batch for smoother decay (optional)
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()
            # Forward pass
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp and (self.device == 'cuda')):
                outputs = self.model(images)
                losses = self.compute_loss(outputs, targets)

            if self.scaler.is_enabled():
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                self.optimizer.step()
            
            # Accumulate losses
            for key, loss in losses.items():
                total_losses[key] += loss.item()
            
            # Store predictions and targets for metrics
            all_predictions['expression'].extend(outputs['expression'].argmax(dim=1).detach().cpu().numpy())
            all_predictions['valence'].extend(outputs['valence'].detach().cpu().numpy())
            all_predictions['arousal'].extend(outputs['arousal'].detach().cpu().numpy())
            
            all_targets['expression'].extend(targets['expression'].cpu().numpy())
            all_targets['valence'].extend(targets['valence'].cpu().numpy())
            all_targets['arousal'].extend(targets['arousal'].cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'expr': f"{losses['expression'].item():.4f}",
                'val': f"{losses['valence'].item():.4f}",
                'aro': f"{losses['arousal'].item():.4f}"
            })

            # Optional fast-limit
            if self.max_train_batches is not None and step >= self.max_train_batches:
                break
        
        # Average losses
        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
        
        # Compute metrics
        metrics = compute_all_metrics(
            all_targets['expression'], all_predictions['expression'],
            y_true_val=all_targets['valence'], y_pred_val=all_predictions['valence'],
            y_true_aro=all_targets['arousal'], y_pred_aro=all_predictions['arousal']
        )
        
        return {**avg_losses, **metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_losses = {'total': 0, 'expression': 0, 'valence': 0, 'arousal': 0}
        all_predictions = {'expression': [], 'valence': [], 'arousal': []}
        all_targets = {'expression': [], 'valence': [], 'arousal': []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for step, batch in enumerate(pbar, start=1):
                # Move to device
                images = batch['images'].to(self.device)
                targets = {
                    'expression': batch['expressions'].to(self.device),
                    'valence': batch['valence'].to(self.device),
                    'arousal': batch['arousal'].to(self.device)
                }
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp and (self.device == 'cuda')):
                    outputs = self.model(images)
                
                # Compute loss
                losses = self.compute_loss(outputs, targets)
                
                # Accumulate losses
                for key, loss in losses.items():
                    total_losses[key] += loss.item()
                
                # Store predictions and targets for metrics
                all_predictions['expression'].extend(outputs['expression'].argmax(dim=1).detach().cpu().numpy())
                all_predictions['valence'].extend(outputs['valence'].detach().cpu().numpy())
                all_predictions['arousal'].extend(outputs['arousal'].detach().cpu().numpy())
                
                all_targets['expression'].extend(targets['expression'].cpu().numpy())
                all_targets['valence'].extend(targets['valence'].cpu().numpy())
                all_targets['arousal'].extend(targets['arousal'].cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'expr': f"{losses['expression'].item():.4f}",
                    'val': f"{losses['valence'].item():.4f}",
                    'aro': f"{losses['arousal'].item():.4f}"
                })

                # Optional fast-limit
                if self.max_val_batches is not None and step >= self.max_val_batches:
                    break
        
        # Average losses
        avg_losses = {k: v / len(self.val_loader) for k, v in total_losses.items()}
        
        # Compute metrics
        metrics = compute_all_metrics(
            all_targets['expression'], all_predictions['expression'],
            y_true_val=all_targets['valence'], y_pred_val=all_predictions['valence'],
            y_true_aro=all_targets['arousal'], y_pred_aro=all_predictions['arousal']
        )
        
        return {**avg_losses, **metrics}
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None) -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        metrics_csv_path: Optional[Path] = None
        if save_dir:
            metrics_csv_path = save_dir / 'metrics.csv'
            if not metrics_csv_path.exists():
                with open(metrics_csv_path, 'w', encoding='utf-8') as f:
                    f.write(
                        'epoch,train_total,train_expr,train_val,train_aro,'
                        'val_total,val_expr,val_val,val_aro,'
                        'expr_acc,expr_f1_macro,expr_f1_weighted,expr_kappa,'
                        'val_rmse,val_mae,val_r2,val_corr,val_sagr,val_ccc,'
                        'aro_rmse,aro_mae,aro_r2,aro_corr,aro_sagr,aro_ccc\n'
                    )
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total'])
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics['total'])
            self.val_metrics.append(val_metrics)
            
            # Print metrics
            print_metrics(train_metrics, "Train")
            print_metrics(val_metrics, "Validation")
            
            # Persist per-epoch metrics
            if metrics_csv_path:
                def mget(d: Dict[str, float], k: str, default: float = 0.0) -> float:
                    return float(d.get(k, default))
                with open(metrics_csv_path, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{epoch+1},"
                        f"{train_metrics['total']:.6f},{train_metrics['expression']:.6f},{train_metrics['valence']:.6f},{train_metrics['arousal']:.6f},"
                        f"{val_metrics['total']:.6f},{val_metrics['expression']:.6f},{val_metrics['valence']:.6f},{val_metrics['arousal']:.6f},"
                        f"{mget(val_metrics,'expr_accuracy'):.6f},{mget(val_metrics,'expr_f1_macro'):.6f},{mget(val_metrics,'expr_f1_weighted'):.6f},{mget(val_metrics,'expr_kappa'):.6f},"
                        f"{mget(val_metrics,'val_rmse'):.6f},{mget(val_metrics,'val_mae'):.6f},{mget(val_metrics,'val_r2'):.6f},{mget(val_metrics,'val_corr'):.6f},{mget(val_metrics,'val_sagr'):.6f},{mget(val_metrics,'val_ccc'):.6f},"
                        f"{mget(val_metrics,'aro_rmse'):.6f},{mget(val_metrics,'aro_mae'):.6f},{mget(val_metrics,'aro_r2'):.6f},{mget(val_metrics,'aro_corr'):.6f},{mget(val_metrics,'aro_sagr'):.6f},{mget(val_metrics,'aro_ccc'):.6f}\n"
                    )

            # Save best model
            if save_dir and val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['total'],
                    'val_metrics': val_metrics
                }, save_dir / 'best_model.pth')
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

