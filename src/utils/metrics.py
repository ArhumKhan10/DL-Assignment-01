"""Evaluation metrics for affect recognition."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, 
    roc_auc_score, average_precision_score,
    mean_squared_error, r2_score
)
import torch
from scipy.stats import pearsonr


def classification_metrics(
    y_true: Union[np.ndarray, List[int]], 
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    num_classes: int = 8
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC metrics)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # AUC metrics (if probabilities provided)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        if y_prob.ndim == 1:
            # Binary case
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        else:
            # Multi-class case
            try:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
                metrics['auc_pr'] = average_precision_score(y_true, y_prob, average='macro')
            except ValueError:
                # Handle case where some classes are missing
                metrics['auc_ovr'] = 0.0
                metrics['auc_ovo'] = 0.0
                metrics['auc_pr'] = 0.0
    
    return metrics


def regression_metrics(
    y_true: Union[np.ndarray, List[float]], 
    y_pred: Union[np.ndarray, List[float]]
) -> Dict[str, float]:
    """Compute regression metrics for valence/arousal prediction.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # Basic regression metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Correlation
    corr, _ = pearsonr(y_true, y_pred)
    metrics['corr'] = corr
    
    # Sign Agreement Metric (SAGR)
    # Penalizes incorrect sign alongside deviation from value
    sign_agreement = np.mean(np.sign(y_true) == np.sign(y_pred))
    metrics['sagr'] = sign_agreement
    
    # Concordance Correlation Coefficient (CCC)
    # Combines Pearson correlation with square difference between means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.cov(y_true, y_pred)[0, 1]
    
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    metrics['ccc'] = ccc
    
    return metrics


def krippendorff_alpha(
    data: np.ndarray,
    level_of_measurement: str = 'nominal'
) -> float:
    """Compute Krippendorff's alpha for inter-rater reliability.
    
    Args:
        data: Array of shape (n_raters, n_items) or (n_items, n_raters)
        level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'
        
    Returns:
        Krippendorff's alpha value
    """
    # Simple implementation - for production use, consider using krippendorff library
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    
    n_raters, n_items = data.shape
    
    if n_raters < 2:
        return 1.0
    
    # Convert to agreement matrix
    unique_values = np.unique(data)
    n_values = len(unique_values)
    
    # Count agreements
    agreements = 0
    total_pairs = 0
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            for rater1 in range(n_raters):
                for rater2 in range(rater1 + 1, n_raters):
                    if data[rater1, i] == data[rater2, j] and data[rater1, j] == data[rater2, i]:
                        agreements += 1
                    total_pairs += 1
    
    if total_pairs == 0:
        return 1.0
    
    alpha = (agreements / total_pairs) if total_pairs > 0 else 0.0
    return alpha


def compute_all_metrics(
    y_true_expr: Union[np.ndarray, List[int]],
    y_pred_expr: Union[np.ndarray, List[int]],
    y_prob_expr: Optional[Union[np.ndarray, List[float]]] = None,
    y_true_val: Optional[Union[np.ndarray, List[float]]] = None,
    y_pred_val: Optional[Union[np.ndarray, List[float]]] = None,
    y_true_aro: Optional[Union[np.ndarray, List[float]]] = None,
    y_pred_aro: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, float]:
    """Compute all metrics for the affect recognition task.
    
    Args:
        y_true_expr: Ground truth expressions
        y_pred_expr: Predicted expressions
        y_prob_expr: Predicted expression probabilities
        y_true_val: Ground truth valence
        y_pred_val: Predicted valence
        y_true_aro: Ground truth arousal
        y_pred_aro: Predicted arousal
        
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    # Classification metrics
    expr_metrics = classification_metrics(y_true_expr, y_pred_expr, y_prob_expr)
    all_metrics.update({f'expr_{k}': v for k, v in expr_metrics.items()})
    
    # Regression metrics for valence
    if y_true_val is not None and y_pred_val is not None:
        val_metrics = regression_metrics(y_true_val, y_pred_val)
        all_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
    
    # Regression metrics for arousal
    if y_true_aro is not None and y_pred_aro is not None:
        aro_metrics = regression_metrics(y_true_aro, y_pred_aro)
        all_metrics.update({f'aro_{k}': v for k, v in aro_metrics.items()})
    
    return all_metrics


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print metrics in a formatted way."""
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    
    # Group metrics by type
    expr_metrics = {k: v for k, v in metrics.items() if k.startswith('expr_')}
    val_metrics = {k: v for k, v in metrics.items() if k.startswith('val_')}
    aro_metrics = {k: v for k, v in metrics.items() if k.startswith('aro_')}
    
    if expr_metrics:
        print("Expression Classification:")
        for k, v in expr_metrics.items():
            print(f"  {k.replace('expr_', '').upper()}: {v:.4f}")
    
    if val_metrics:
        print("\nValence Regression:")
        for k, v in val_metrics.items():
            print(f"  {k.replace('val_', '').upper()}: {v:.4f}")
    
    if aro_metrics:
        print("\nArousal Regression:")
        for k, v in aro_metrics.items():
            print(f"  {k.replace('aro_', '').upper()}: {v:.4f}")

