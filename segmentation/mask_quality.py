"""
Combine UNet and SAM masks for the clinical close-ups: per-image agreement (Dice), quality flag, and
visual QA contact sheets (random samples per flag, overlay of UNet in green and SAM in magenta).

    python segmentation/mask_quality.py --images uncertaintyNet-main/datasets_768/ISIC_clinical \
        --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks --out segmentation/output/qa

Feature table v1 uses the clinical fine-tuned UNet and its gold-calibrated flag (postprocess.quality_flag_v1):

    python segmentation/mask_quality.py --images uncertaintyNet-main/datasets_768/ISIC_clinical \
        --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks --primary unet_clinical --flag v1 \
        --split isic_clinical_v2.csv --out segmentation/output/qa_clinical

The primary mask's statistics keep the `unet_*` column names (the feature table schema); `mask_source` says
which model they came from.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from postprocess import dice, quality_flag, quality_flag_v1

ROOT = Path(__file__).resolve().parents[1]


def outline(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def overlay(img, unet, sam, size=192):
    out = img.copy()
    out[outline(sam)] = (255, 0, 255)
    out[outline(unet)] = (0, 255, 0)
    im = Image.fromarray(out)
    im.thumbnail((size, size))
    tile = Image.new("RGB", (size, size), "black")
    tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return tile


def contact_sheet(rows, images, masks, path, n=48, cols=8, seed=0, primary="unet"):
    rows = rows.sample(min(n, len(rows)), random_state=seed) if len(rows) else rows
    size = 192
    sheet = Image.new("RGB", (cols * size, ((len(rows) + cols - 1) // cols) * (size + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows.itertuples()):
        img = np.array(Image.open(images / f"{r.isic_id}.jpg").convert("RGB"))
        unet = np.array(Image.open(masks / primary / f"{r.isic_id}.png")) > 127
        sam = np.array(Image.open(masks / "sam" / f"{r.isic_id}.png")) > 127
        x, y = (k % cols) * size, (k // cols) * (size + 14)
        sheet.paste(overlay(img, unet, sam, size), (x, y))
        draw.text((x + 2, y + size), f"{r.isic_id} d={r.unet_sam_dice:.2f}", fill="black")
    sheet.save(path)


def main(opts):
    images, masks, out = Path(opts.images), Path(opts.masks), Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    unet = pd.read_csv(masks / opts.primary / "unet_stats.csv")
    sam = pd.read_csv(masks / "sam" / "sam_stats.csv")
    df = unet.merge(sam, on="isic_id", how="left")
    df["unet_sam_dice"] = [
        dice(np.array(Image.open(masks / opts.primary / f"{i}.png")) > 127,
             np.array(Image.open(masks / "sam" / f"{i}.png")) > 127)
        for i in df.isic_id]
    if opts.flag == "v1":
        df["mask_quality"] = [quality_flag_v1(a, b, d, p) for a, b, d, p in zip(
            df.unet_area_ratio, df.unet_border_touch, df.unet_sam_dice, df.unet_mean_prob_in_mask)]
    else:
        df["mask_quality"] = [quality_flag(a, b, d) for a, b, d in
                              zip(df.unet_area_ratio, df.unet_border_touch, df.unet_sam_dice)]
    df.insert(1, "mask_source", opts.primary)
    df.insert(2, "mask_quality_version", opts.flag)
    split = pd.read_csv(ROOT / "splits" / opts.split)[["isic_id", "split", "label", "attribution"]]
    df = df.merge(split, on="isic_id", how="left")
    df.to_csv(out / "mask_quality.csv", index=False)

    for flag in ("ok", "review", "fail"):
        contact_sheet(df[df.mask_quality == flag], images, masks, out / f"sheet_{flag}.png", primary=opts.primary)
    contact_sheet(df.nsmallest(48, "unet_sam_dice"), images, masks, out / "sheet_lowest_agreement.png",
                  primary=opts.primary)

    summary = {
        "n": len(df),
        "flag_counts": df.mask_quality.value_counts().to_dict(),
        "unet_sam_dice_median": float(df.unet_sam_dice.median()),
        "unet_sam_dice_by_source": df.groupby("attribution").unet_sam_dice.median().round(3).to_dict(),
        "flag_by_source": pd.crosstab(df.attribution, df.mask_quality).to_dict(),
        "flag_by_label": pd.crosstab(df.label, df.mask_quality).to_dict(),
    }
    pd.Series(summary).to_json(out / "summary.json", indent=1)
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary", default="unet", help="mask folder under --masks whose stats and flag are reported")
    ap.add_argument("--flag", choices=["v0", "v1"], default="v0", help="v0: original UNet rule; v1: gold-calibrated")
    ap.add_argument("--split", default="isic_clinical_v1.csv", help="file name under splits/")
    main(ap.parse_args())
