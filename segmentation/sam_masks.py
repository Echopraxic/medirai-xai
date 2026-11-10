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


def unet_box(unet_mask: np.ndarray, margin: float = 0.1):
    """Bounding box (x0, y0, x1, y1) of a UNet mask, grown by `margin` of its size on each side and clipped to the
    image; None for an empty mask or one covering most of the frame (a failed mask is no prompt)."""
    ys, xs = np.nonzero(unet_mask)
    if len(ys) == 0 or unet_mask.mean() > 0.85:
        return None
    h, w = unet_mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    dy, dx = margin * (y1 - y0 + 1), margin * (x1 - x0 + 1)
    return np.array([max(0, x0 - dx), max(0, y0 - dy), min(w - 1, x1 + dx), min(h - 1, y1 + dy)], dtype=float)


def box_prompt_mask(predictor: SamPredictor, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, float]:
    """SAM with the UNet's box plus a positive point at its centre; single output (a box is unambiguous)."""
    predictor.set_image(img)
    centre = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
    masks, scores, _ = predictor.predict(point_coords=centre, point_labels=np.array([1]), box=box,
                                         multimask_output=False)
    return masks[0], float(scores[0])


def main(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[opts.model](checkpoint=opts.ckpt).to(device).eval()
    predictor = SamPredictor(sam)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    if opts.kind == "clinical":
        split = pd.read_csv(ROOT / "splits" / "isic_clinical_v1.csv")
        ids = split.loc[split.split != "excluded", "isic_id"]
        if opts.ids:
            ids = ids[ids.isin(set(pd.read_csv(ROOT / opts.ids).isic_id))]
        items = [(i, Path(opts.images) / f"{i}.jpg", None) for i in ids]
    else:
        imgs = sorted((Path(opts.images) / "img").glob("*.jpg"))
        items = [(p.stem, p, Path(opts.images) / "seg" / f"{p.stem}_segmentation.png") for p in imgs]

    rows = []
    with torch.no_grad():
        for n, (key, path, gt_path) in enumerate(items, 1):
            img = np.array(Image.open(path).convert("RGB"))
            box = None
            if opts.unet_masks and (Path(opts.unet_masks) / f"{key}.png").exists():
                box = unet_box(np.array(Image.open(Path(opts.unet_masks) / f"{key}.png")) > 127)
            raw, score = box_prompt_mask(predictor, img, box) if box is not None else centre_point_mask(predictor, img)
            mask, n_comp = clean_mask(raw)
            row = {"isic_id": key, "sam_score": score, "sam_area_ratio": float(mask.mean()), "sam_components": n_comp,
                   "sam_prompt": "unet_box" if box is not None else "centre"}
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
    ap.add_argument("--model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="must match --ckpt")
    ap.add_argument("--unet_masks", default="",
                    help="clinical: prompt SAM with each UNet mask's box (+ its centre) instead of the image centre; "
                         "images with no usable UNet mask fall back to the centre point")
    ap.add_argument("--ids", default="", help="clinical: csv under the repo with an isic_id column (e.g. the gold set)")
    main(ap.parse_args())
