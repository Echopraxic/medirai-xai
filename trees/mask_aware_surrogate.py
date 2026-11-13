"""
Mask-aware surrogate prototype (validation only; no test or locked set is read).

Failed lesion masks are mostly malignant images (split v2 val: 84% of `fail` vs 53% of `ok`), so a surrogate on
mask-based features can learn "features of a failed mask = malignant" and then contradict the ResNet's benign calls on
those images (kdv8f_mel s3, features v1.1: 56.8% agreement on failed-mask benign calls vs 85.8% with an ok mask).
This trains the surrogate (ResNet call as target, trees/train_trees.fit_tuned) with and without the mask's quality
signals, so the tree can tell a failed outline from an irregular lesion, and reports val agreement per class and per
mask flag.

Arms (features v1.0 table; the MONET concepts of v1.1 are not in it, so compare arms with each other, not with the
v1.1 surrogate):
  base     handcrafted features only (as the current surrogate)
  flag     + one-hot mask flag (fail / review)
  quality  + flag + the UNet quality statistics behind it (mean probability in mask, SAM agreement, area ratio,
           border touch, boundary uncertainty)

    python trees/mask_aware_surrogate.py --features features/output/feature_table_v1.0.parquet \
        --calls trees/output_v1.1_kdv8f_mel_s3/tree_predictions.csv --out trees/output/mask_aware_surrogate.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_trees import feature_columns, fit_tuned

QUALITY = ["unet_mean_prob_in_mask", "unet_sam_dice", "unet_area_ratio", "unet_border_touch",
           "unet_boundary_uncertainty"]


def arm_columns(df, arm):
    cols = feature_columns(df)
    if arm in ("flag", "quality"):
        cols = cols + ["mask_fail", "mask_review"]
    if arm == "quality":
        cols = cols + QUALITY
    return cols


def agreement(d):
    """Overall and per-class agreement with the ResNet, overall and by mask flag."""
    a = d.pred == d.target
    out = {"overall": float(a.mean()), "n": len(d)}
    for k, name in ((0, "benign_calls"), (1, "malignant_calls")):
        out[name] = float(a[d.target == k].mean())
    out["lower_class"] = min(out["benign_calls"], out["malignant_calls"])
    out["by_mask"] = {q: {"benign_calls": float(a[(d.mask_quality == q) & (d.target == 0)].mean()),
                          "malignant_calls": float(a[(d.mask_quality == q) & (d.target == 1)].mean()),
                          "n_benign_calls": int(((d.mask_quality == q) & (d.target == 0)).sum())}
                      for q in ("ok", "review", "fail")}
    return out


def main(opts):
    f = pd.read_parquet(opts.features)
    calls = pd.read_csv(opts.calls)[["isic_id", "resnet_pred"]]
    d = f.merge(calls, on="isic_id").rename(columns={"resnet_pred": "target"})
    d = d[d.extraction_error.fillna("") == ""].reset_index(drop=True)
    d["mask_fail"] = (d.mask_quality == "fail").astype(int)
    d["mask_review"] = (d.mask_quality == "review").astype(int)
    tr, va = d[d.split == "train"], d[d.split == "val"].copy()
    res = {"n_train": len(tr), "n_val": len(va),
           "val_malignant_share_by_mask": va.groupby("mask_quality").label.mean().round(3).to_dict(), "arms": {}}
    for arm in ("base", "flag", "quality"):
        cols = arm_columns(d, arm)
        model, params, _ = fit_tuned(tr[cols], tr.target, va[cols], va.target)
        va["pred"] = (model.predict_proba(va[cols])[:, 1] >= 0.5).astype(int)
        res["arms"][arm] = {"params": params, "val": agreement(va)}
        r = res["arms"][arm]["val"]
        print(f"{arm:8s} overall {r['overall']:.3f}  benign {r['benign_calls']:.3f}  malignant {r['malignant_calls']:.3f}"
              f"  | fail-mask benign {r['by_mask']['fail']['benign_calls']:.3f}"
              f"  ok-mask benign {r['by_mask']['ok']['benign_calls']:.3f}", flush=True)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--calls", type=Path, required=True, help="tree_predictions.csv with resnet_pred per image")
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args())
