import glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "Dataset" / "annotations"
IMG = ROOT / "Dataset" / "images"

def main():
    ids = sorted([Path(p).name.replace("_exp.npy","") for p in glob.glob(str(ANN / "*_exp.npy"))])
    print(f"Found {len(ids)} ids. First 5: {ids[:5]}")
    if not ids:
        return
    sid = ids[0]
    exp = np.load(ANN / f"{sid}_exp.npy")
    val = np.load(ANN / f"{sid}_val.npy")
    aro = np.load(ANN / f"{sid}_aro.npy")
    lnd = np.load(ANN / f"{sid}_lnd.npy")
    print("Sample shapes:", {
        "exp": getattr(exp, 'shape', None),
        "val": getattr(val, 'shape', None),
        "aro": getattr(aro, 'shape', None),
        "lnd": getattr(lnd, 'shape', None),
    })
    print("Sample values:", {
        "exp": exp if exp.shape == () else exp[:5],
        "val": val if val.shape == () else val[:5],
        "aro": aro if aro.shape == () else aro[:5],
        "lnd": lnd[:2]
    })

if __name__ == "__main__":
    main()
