from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allow inspection without torch
    class Dataset:  # type: ignore
        pass

@dataclass
class AffectItem:
    image_path: Path
    expression: int
    valence: float
    arousal: float
    landmarks: np.ndarray

class AffectDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        return_pil: bool = False,
        exclude_uncertain: bool = True,
        ids: Optional[list[str]] = None,
    ) -> None:
        self.root = Path(root_dir)
        self.ann = self.root / "Dataset" / "annotations"
        self.img = self.root / "Dataset" / "images"
        self.transform = transform
        self.return_pil = return_pil
        self.exclude_uncertain = exclude_uncertain

        if ids is None:
            ids = sorted([p.name.replace("_exp.npy", "") for p in self.ann.glob("*_exp.npy")])
        self.ids = []
        self._index: list[AffectItem] = []

        for sid in ids:
            img_path = self.img / f"{sid}.jpg"
            exp_path = self.ann / f"{sid}_exp.npy"
            val_path = self.ann / f"{sid}_val.npy"
            aro_path = self.ann / f"{sid}_aro.npy"
            lnd_path = self.ann / f"{sid}_lnd.npy"
            if not (img_path.exists() and exp_path.exists() and val_path.exists() and aro_path.exists() and lnd_path.exists()):
                continue
            try:
                exp = int(np.load(exp_path))
                val = float(np.load(val_path))
                aro = float(np.load(aro_path))
                lnd = np.load(lnd_path)
            except Exception:
                continue
            if self.exclude_uncertain and (val == -2 or aro == -2):
                continue
            self.ids.append(sid)
            self._index.append(
                AffectItem(
                    image_path=img_path,
                    expression=exp,
                    valence=val,
                    arousal=aro,
                    landmarks=lnd,
                )
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self._index[idx]
        img = Image.open(item.image_path).convert("RGB")
        sample: Dict[str, object] = {
            "image": img if self.return_pil or self.transform is None else None,
            "expression": item.expression,
            "valence": item.valence,
            "arousal": item.arousal,
            "landmarks": item.landmarks,
            "id": self.ids[idx],
        }
        if self.transform is not None:
            sample["image"] = self.transform(img)
        return sample
