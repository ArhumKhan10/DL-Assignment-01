"""Entry script to sanity-check dataloaders and model wiring."""

from __future__ import annotations

from pathlib import Path
import argparse
import torch

from ..data.loaders import get_dataloaders
from ..models.baseline_models import create_resnet_model, create_efficientnet_model
from .trainer import AffectTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run affect training")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "efficientnet"], help="Backbone model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fast", action="store_true", help="Fast mode: limit batches for quick sanity")
    parser.add_argument("--out", type=str, default="outputs/run", help="Output directory for checkpoints and metrics")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights for the backbone")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for DataLoader")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) training on CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    splits_path = root / "data" / "splits.json"

    # Create loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=root,
        splits_path=splits_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    print(
        f"Loaders ready: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    # Pick backbone (toggle between 'resnet' and 'efficientnet')
    if args.backbone == "resnet":
        model = create_resnet_model(pretrained=args.pretrained)
    else:
        model = create_efficientnet_model(pretrained=args.pretrained)

    # Fast 1-epoch sanity training
    trainer = AffectTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=2e-4,
        expression_weight=1.0,
        valence_weight=1.0,
        arousal_weight=1.0,
        max_train_batches=(5 if args.fast else None),
        max_val_batches=(2 if args.fast else None),
        use_amp=args.amp,
    )

    save_dir = Path(args.out)
    history = trainer.train(num_epochs=args.epochs, save_dir=save_dir)
    print("Done. Last val loss:", history["val_losses"][-1])


if __name__ == "__main__":
    main()


