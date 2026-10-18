"""
Our handcrafted features on the Derm7pt CLINICAL photos, for comparison with the Derm7pt expert labels and
with VLM concept answers (concepts/eval_vlm_concepts.py). Masks come from the clinical fine-tuned UNet
(segmentation/finetune_clinical.py); Derm7pt has no lesion masks.

    python concepts/derm7pt_features.py --ckpt segmentation/output/unet_r18_clinical/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "segmentation"))
import extract as fx  # noqa: E402
from postprocess import border_touch_fraction, clean_mask  # noqa: E402
from predict_unet import predict_prob  # noqa: E402
from unet_resnet import ResNetUNet  # noqa: E402


def main(opts):
    root = Path(opts.derm7pt)
    meta = pd.read_csv(root / "meta" / "meta.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(opts.ckpt, map_location=device)
    net = ResNetUNet(pretrained=False).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    mask_dir = ROOT / "concepts/output/derm7pt_clinic_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, r in enumerate(meta.itertuples(), 1):
        img = Image.open(root / "images" / r.clinic).convert("RGB")
        img.thumbnail((768, 768))                       # same working cache size as the ISIC clinical images
        mask, _ = clean_mask(predict_prob(net, img, ckpt["size"], device) > 0.5)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / f"{r.case_num}.png")
        row = {"case_num": r.case_num, "mask_area_ratio": float(mask.mean()), "mask_border_touch": border_touch_fraction(mask)}
        try:
            row.update(fx.extract(np.array(img), mask))
        except Exception as err:
            row["extraction_error"] = f"{type(err).__name__}: {err}"
        rows.append(row)
        if n % 200 == 0:
            print(f"{n}/{len(meta)}", flush=True)
    df = pd.DataFrame(rows)
    df["feature_version"] = fx.FEATURE_VERSION
    df.to_parquet(ROOT / "concepts/output/derm7pt_clinic_features.parquet", index=False)
    print(f"done: {len(df)} rows, {df.get('extraction_error', pd.Series(dtype=str)).notna().sum()} errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "segmentation/output/unet_r18_clinical/best.pt"))
    ap.add_argument("--derm7pt", default=str(ROOT / "data/derm7pt/release_v0"))
    main(ap.parse_args())
