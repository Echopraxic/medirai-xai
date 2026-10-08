"""
Score mask methods against the hand-annotated gold set (segmentation/gold/masks) and check the quality flag.

For each method directory (PNG masks named <isic_id>.png at the cached image size): Dice and IoU per gold
image, summarized by source, by current flag and by half (tune/report). Then: does the ok/review/fail flag
(and UNet-SAM agreement) predict a bad mask (gold Dice < 0.8)? Choose methods and flag thresholds on the
`tune` half; quote numbers from the `report` half.

    python segmentation/gold_eval.py --methods unet=uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet \
        sam=uncertaintyNet-main/datasets_768/ISIC_clinical_masks/sam \
        unet_clinical=uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from postprocess import dice

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "segmentation/gold"
BAD_DICE = 0.8


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def main(opts):
    gold_set = pd.read_csv(GOLD / "gold_set.csv")
    done = {p.stem for p in (GOLD / "masks").glob("*.png")}
    gold_set = gold_set[gold_set.isic_id.isin(done)].reset_index(drop=True)
    if gold_set.empty:
        raise SystemExit("No gold masks yet: run segmentation/annotate.py first.")
    methods = dict(m.split("=", 1) for m in opts.methods)
    rows = []
    for r in gold_set.itertuples():
        g = np.array(Image.open(GOLD / "masks" / f"{r.isic_id}.png")) > 127
        for name, folder in methods.items():
            path = ROOT / folder / f"{r.isic_id}.png"
            if not path.exists():
                continue
            m = np.array(Image.open(path)) > 127
            if m.shape != g.shape:
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(g.shape[::-1], Image.NEAREST)) > 127
            rows.append({"isic_id": r.isic_id, "method": name, "source": r.source, "label": r.label,
                         "flag": r.mask_quality, "half": r.half, "unet_sam_dice": r.unet_sam_dice,
                         "dice": dice(m, g), "iou": iou(m, g)})
    df = pd.DataFrame(rows)
    out = ROOT / "segmentation/output/gold_eval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_image.csv", index=False)

    agg = lambda d: d.groupby("method").agg(n=("dice", "size"), dice_mean=("dice", "mean"),  # noqa: E731
                                             dice_median=("dice", "median"),
                                             bad_rate=("dice", lambda x: float((x < BAD_DICE).mean())))
    report = {"n_gold": len(gold_set), "bad_dice_threshold": BAD_DICE,
              "overall_by_half": {h: agg(df[df.half == h]).round(3).to_dict("index") for h in ("tune", "report")},
              "by_source": {s: agg(df[df.source == s]).round(3).to_dict("index") for s in sorted(df.source.unique())},
              "by_flag": {f: agg(df[df.flag == f]).round(3).to_dict("index") for f in ("ok", "review", "fail")}}
    unet = df[df.method == "unet"]
    if len(unet) and unet.dice.lt(BAD_DICE).nunique() == 2:
        bad = unet.dice < BAD_DICE
        report["flag_check_unet"] = {
            "auroc_agreement_predicts_bad": float(roc_auc_score(bad, -unet.unet_sam_dice)),
            "bad_rate_by_flag": unet.groupby("flag").dice.apply(lambda x: float((x < BAD_DICE).mean())).round(3).to_dict()}
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, help="name=folder (relative to the repo root)")
    main(ap.parse_args())
