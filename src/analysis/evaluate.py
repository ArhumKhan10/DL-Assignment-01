"""Evaluation utilities: confusion matrices, ROC/PR curves, test metrics, error grids.

Outputs (under outputs/comparison/):
  - confusion_<model>_<split>.png
  - roc_pr_<model>_<split>.png
  - test_metrics.csv (aggregated)
  - errors_<model>.png (misclassified + high regression error)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import pandas as pd

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.loaders import get_dataloaders
from models.baseline_models import create_resnet_model, create_efficientnet_model
from utils.metrics import regression_metrics


NUM_CLASSES = 8


def _softmax_logits(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=1).detach().cpu().numpy()


def _collect_split(model: torch.nn.Module, loader: DataLoader) -> Dict[str, np.ndarray | List[str]]:
    model.eval()
    y_true_cls: List[int] = []
    y_prob_cls: List[np.ndarray] = []
    y_pred_cls: List[int] = []
    y_true_val: List[float] = []
    y_pred_val: List[float] = []
    y_true_aro: List[float] = []
    y_pred_aro: List[float] = []
    ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch["images"])  # dict
            probs = _softmax_logits(out["expression"])  # (B, C)
            preds = probs.argmax(axis=1)

            y_true_cls.extend(batch["expressions"].cpu().numpy().tolist())
            y_prob_cls.extend(probs)
            y_pred_cls.extend(preds.tolist())
            y_true_val.extend(batch["valence"].cpu().numpy().tolist())
            y_true_aro.extend(batch["arousal"].cpu().numpy().tolist())
            y_pred_val.extend(out["valence"].detach().cpu().numpy().tolist())
            y_pred_aro.extend(out["arousal"].detach().cpu().numpy().tolist())
            ids.extend(batch["ids"])

    return {
        "y_true_cls": np.array(y_true_cls),
        "y_prob_cls": np.vstack(y_prob_cls),
        "y_pred_cls": np.array(y_pred_cls),
        "y_true_val": np.array(y_true_val),
        "y_pred_val": np.array(y_pred_val),
        "y_true_aro": np.array(y_true_aro),
        "y_pred_aro": np.array(y_pred_aro),
        "ids": ids,
    }


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticklabels(range(NUM_CLASSES)); ax.set_yticklabels(range(NUM_CLASSES))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path) -> None:
    # y_true: (N,) labels; y_prob: (N, C)
    y_true_b = np.eye(NUM_CLASSES)[y_true]
    # ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # ROC curves
    roc_aucs = []
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_b[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        axes[0].plot(fpr, tpr, lw=1, label=f"Class {c} (AUC={roc_auc:.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title(f"ROC - {title}")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    # PR curves
    for c in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true_b[:, c], y_prob[:, c])
        ap = average_precision_score(y_true_b[:, c], y_prob[:, c])
        axes[1].plot(recall, precision, lw=1, label=f"Class {c} (AP={ap:.2f})")
    axes[1].set_title(f"PR - {title}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_all(models_to_run: List[str], splits_to_run: List[str], batch_size: int) -> None:
    root = Path(__file__).resolve().parents[2]
    comp_dir = root / "outputs" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    splits_path = root / "data" / "splits.json"
    # Build loaders (same transforms as val/test)
    _, val_loader, test_loader = get_dataloaders(root, splits_path, batch_size=batch_size, num_workers=0)

    # Prepare models
    models: Dict[str, torch.nn.Module] = {}
    paths = {
        "resnet50": root / "outputs" / "resnet50_e34" / "best_model.pth",
        "efficientnet_b0": root / "outputs" / "efficientnet_e34" / "best_model.pth",
        "resnet50_v2": root / "outputs" / "resnet50_v2" / "best_model.pth",
    }
    for name, ckpt in paths.items():
        if not ckpt.exists():
            continue
        if models_to_run and name not in models_to_run:
            continue
        if name.startswith("resnet"):
            m = create_resnet_model(pretrained=False)
        else:
            m = create_efficientnet_model(pretrained=False)
        # Torch 2.6 default weights_only=True; allow full load for our checkpoints
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(state["model_state_dict"])
        models[name] = m

    rows = []
    for model_name, model in models.items():
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            if splits_to_run and split_name not in splits_to_run:
                continue
            data = _collect_split(model, loader)
            y_true = data["y_true_cls"].astype(int)
            y_pred = data["y_pred_cls"].astype(int)
            y_prob = data["y_prob_cls"].astype(float)
            # Confusion
            _plot_confusion(y_true, y_pred, f"{model_name} - {split_name}", comp_dir / f"confusion_{model_name}_{split_name}.png")
            # ROC/PR
            _plot_roc_pr(y_true, y_prob, f"{model_name} - {split_name}", comp_dir / f"roc_pr_{model_name}_{split_name}.png")
            # Per-split metrics summary
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            expr_acc = float(cls_report["accuracy"]) if "accuracy" in cls_report else float((y_true == y_pred).mean())
            # Regression
            val_metrics = regression_metrics(data["y_true_val"], data["y_pred_val"])
            aro_metrics = regression_metrics(data["y_true_aro"], data["y_pred_aro"])
            rows.append({
                "model": model_name,
                "split": split_name,
                "expr_accuracy": expr_acc,
                "expr_f1_macro": float(cls_report.get("macro avg", {}).get("f1-score", 0.0)),
                "val_rmse": float(val_metrics["rmse"]),
                "val_ccc": float(val_metrics["ccc"]),
                "aro_rmse": float(aro_metrics["rmse"]),
                "aro_ccc": float(aro_metrics["ccc"]),
            })
            # Error grids
            # Misclassified examples
            wrong_idx = np.where(y_true != y_pred)[0]
            # Sort by abs valence error
            val_err = np.abs(data["y_true_val"] - data["y_pred_val"])
            worst_val_idx = np.argsort(-val_err)[:8]
            # We cannot re-render images easily here because loaders return tensors post-normalize.
            # As an alternative, save IDs lists for manual inspection.
            with open(comp_dir / f"errors_{model_name}_{split_name}.txt", "w", encoding="utf-8") as f:
                f.write("Misclassified IDs (first 32):\n")
                for idx in wrong_idx[:32]:
                    f.write(f"{data['ids'][idx]}: true={int(y_true[idx])}, pred={int(y_pred[idx])}\n")
                f.write("\nWorst valence error IDs (top 32):\n")
                for idx in worst_val_idx[:32]:
                    f.write(f"{data['ids'][idx]}: true={float(data['y_true_val'][idx]):.3f}, pred={float(data['y_pred_val'][idx]):.3f}\n")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(comp_dir / "test_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate eval artifacts (confusion, ROC/PR, metrics).")
    p.add_argument("--models", nargs="*", default=[], help="Subset: resnet50 efficientnet_b0 resnet50_v2")
    p.add_argument("--splits", nargs="*", default=["val", "test"], help="Splits to run: val test")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args.models, args.splits, args.batch_size)


