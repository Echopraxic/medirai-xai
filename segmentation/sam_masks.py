"""
Zero-shot SAM (ViT-B) lesion masks from a single positive point at the image centre, as an independent
second opinion on the UNet. Of SAM's three candidate masks, keep the highest-scoring one whose area is a
plausible lesion size (0.2%-85% of the image).

Assumes the lesion is near the image centre (true for most close-ups); off-centre lesions show up as
UNet/SAM disagreement and are flagged downstream.

    # clinical close-ups (no ground truth)
    python segmentation/sam_masks.py clinical --images uncertaintyNet-main/datasets_768/ISIC_clinical \
        --out uncertaintyNet-main/datasets_768/ISIC_clinical_masks/sam --ckpt uncertaintyNet-main/datasets/sam/sam_vit_b_01ec64.pth
    # ISIC 2018 test (ground truth) -> Dice/Jaccard comparable to the UNet's
    python segmentation/sam_masks.py isic2018 --images uncertaintyNet-main/datasets_768/ISIC2018/test \
        --out segmentation/output/sam_isic2018_test --ckpt ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from postprocess import clean_mask, dice

ROOT = Path(__file__).resolve().parents[1]


def centre_point_mask(predictor: SamPredictor, img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(point_coords=np.array([[w / 2, h / 2]]), point_labels=np.array([1]),
                                         multimask_output=True)
    order = np.argsort(-scores)
    for k in order:
        ratio = masks[k].mean()
        if 0.002 <= ratio <= 0.85:
            return masks[k], float(scores[k])
    return masks[order[0]], float(scores[order[0]])


def main(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=opts.ckpt).to(device).eval()
    predictor = SamPredictor(sam)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    if opts.kind == "clinical":
        split = pd.read_csv(ROOT / "splits" / "isic_clinical_v1.csv")
        items = [(i, Path(opts.images) / f"{i}.jpg", None) for i in split.loc[split.split != "excluded", "isic_id"]]
    else:
        imgs = sorted((Path(opts.images) / "img").glob("*.jpg"))
        items = [(p.stem, p, Path(opts.images) / "seg" / f"{p.stem}_segmentation.png") for p in imgs]

    rows = []
    with torch.no_grad():
        for n, (key, path, gt_path) in enumerate(items, 1):
            img = np.array(Image.open(path).convert("RGB"))
            raw, score = centre_point_mask(predictor, img)
            mask, n_comp = clean_mask(raw)
            row = {"isic_id": key, "sam_score": score, "sam_area_ratio": float(mask.mean()), "sam_components": n_comp}
            if gt_path is not None:
                gt = np.array(Image.open(gt_path)) > 127
                inter, union = (mask & gt).sum(), (mask | gt).sum()
                row.update(dice=dice(mask, gt), jaccard=float(inter / union) if union else 1.0)
            else:
                Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{key}.png")
            rows.append(row)
            if n % 500 == 0:
                print(f"{n}/{len(items)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sam_stats.csv", index=False)
    if opts.kind == "isic2018":
        j = df["jaccard"]
        summary = {"dice": df["dice"].mean(), "jaccard": j.mean(), "thresholded_jaccard": j.where(j >= 0.65, 0).mean(),
                   "n": len(df)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print("TEST", json.dumps(summary))
    print(f"done: {len(df)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["clinical", "isic2018"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    main(ap.parse_args())
