"""Single-image inference for expression + valence/arousal.

Usage:
  .\.venv\Scripts\Activate.ps1
  python scripts/infer_image.py --model outputs/resnet50_e34/best_model.pth --backbone resnet --image path\to\image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.baseline_models import create_resnet_model, create_efficientnet_model
from src.data.transforms import get_val_transforms


def load_model(backbone: str, ckpt_path: Path) -> torch.nn.Module:
    if backbone == "resnet":
        model = create_resnet_model(pretrained=False)
    elif backbone == "efficientnet":
        model = create_efficientnet_model(pretrained=False)
    else:
        raise ValueError("backbone must be 'resnet' or 'efficientnet'")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def predict_image(model: torch.nn.Module, image_path: Path) -> dict:
    transform = get_val_transforms()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1,3,H,W)
    with torch.no_grad():
        out = model(x)
        logits = out["expression"]
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_cls = int(probs.argmax())
        val = float(out["valence"][0].cpu().numpy())
        aro = float(out["arousal"][0].cpu().numpy())
    return {"pred_class": pred_cls, "probs": probs.tolist(), "valence": val, "arousal": aro}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best_model.pth")
    ap.add_argument("--backbone", default="resnet", choices=["resnet", "efficientnet"], help="Backbone type")
    ap.add_argument("--image", required=True, help="Path to image (jpg/png)")
    args = ap.parse_args()

    result = predict_image(load_model(args.backbone, Path(args.model)), Path(args.image))
    print(result)


if __name__ == "__main__":
    main()


