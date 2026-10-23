"""
Handcrafted features (and flat copies for the MONET embedding step) for concept-validation images that sit in one flat
folder of <id>.jpg files: the SkinCon-labelled Fitzpatrick17k neoplastic images, and the Derm7pt clinical photos
(prereg 2026-10-23, section A). Same masks and extractor as the Derm7pt run (clinical fine-tuned UNet,
features.extract), so the numbers are comparable. Read-only on the images; outputs go to concepts/output/.

    python concepts/skincon_features.py skincon --images data/fitzpatrick17k/images_skincon_neoplastic
    python concepts/skincon_features.py derm7pt-flat       # data/derm7pt/clinic_flat/<case_num>.jpg (<= 768 px)
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


def skincon(opts):
    ann = pd.read_csv(ROOT / "data/skincon/annotations_fitzpatrick17k.csv", index_col=0)
    ann["md5hash"] = ann.ImageID.str.replace(".jpg", "", regex=False)
    f17k = pd.read_csv(ROOT / "data/fitzpatrick17k/fitzpatrick17k.csv")
    ids = ann.merge(f17k[["md5hash", "three_partition_label"]], on="md5hash").query(
        "three_partition_label in ['benign', 'malignant']").md5hash.tolist()
    folder = Path(opts.images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(opts.ckpt, map_location=device)
    net = ResNetUNet(pretrained=False).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    rows = []
    for n, i in enumerate(ids, 1):
        img = Image.open(folder / f"{i}.jpg").convert("RGB")
        img.thumbnail((768, 768))
        mask, _ = clean_mask(predict_prob(net, img, ckpt["size"], device) > 0.5)
        row = {"md5hash": i, "mask_area_ratio": float(mask.mean()), "mask_border_touch": border_touch_fraction(mask)}
        try:
            row.update(fx.extract(np.array(img), mask))
        except Exception as err:
            row["extraction_error"] = f"{type(err).__name__}: {err}"
        rows.append(row)
        if n % 200 == 0:
            print(f"{n}/{len(ids)}", flush=True)
    df = pd.DataFrame(rows)
    df["feature_version"] = fx.FEATURE_VERSION
    df.to_parquet(ROOT / "concepts/output/skincon_f17k_features.parquet", index=False)
    print(f"done: {len(df)} rows, {df.get('extraction_error', pd.Series(dtype=str)).notna().sum()} errors")


def derm7pt_flat(opts):
    root = ROOT / "data/derm7pt/release_v0"
    meta = pd.read_csv(root / "meta/meta.csv")
    out = ROOT / "data/derm7pt/clinic_flat"
    out.mkdir(exist_ok=True)
    for r in meta.itertuples():
        img = Image.open(root / "images" / r.clinic).convert("RGB")
        img.thumbnail((768, 768))
        img.save(out / f"{r.case_num}.jpg", quality=95)
    pd.DataFrame({"isic_id": meta.case_num, "split": "all"}).to_csv(out / "index.csv", index=False)
    print(f"{len(meta)} flat copies in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["skincon", "derm7pt-flat"])
    ap.add_argument("--images", default=str(ROOT / "data/fitzpatrick17k/images_skincon_neoplastic"))
    ap.add_argument("--ckpt", default=str(ROOT / "segmentation/output/unet_r18_clinical/best.pt"))
    a = ap.parse_args()
    {"skincon": skincon, "derm7pt-flat": derm7pt_flat}[a.step](a)
