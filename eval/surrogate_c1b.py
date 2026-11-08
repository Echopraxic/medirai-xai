"""
C1 out-of-sample targets (plan decision 2026-11-04, fixed before any of these predictions existed). Final model
kdv8f_mel s3, features v1.1, the current tree's recipe (hard target, no mask-fail rows, all features, same grid).

Arms: current; +E (SD-198 + Mendeley melanoma images the final model never trained on, as extra train rows);
O (split-v2 train rows learn the out-of-fold ResNet call from 5 fold models instead of the in-sample call); O+E.
Adopt the best arm by the lower per-class val agreement only if: val score >= current + 0.01; overall val agreement
>= current - 0.005; and on test (one look) overall and benign agreement each >= current - 0.005.
Extra rows have no SAM mask, so their poor-mask flag is quality_flag_v1 without the SAM-agreement term.

    python eval/surrogate_c1b.py   -> eval/output/surrogate_c1/report_c1b.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "segmentation"))
from eval.surrogate_c1 import FEATURES, RUN, scores  # noqa: E402
from features.add_monet import MONET_FEATURES  # noqa: E402
from trees.train_trees import feature_columns, fit_tuned  # noqa: E402

OUT = ROOT / "eval/output/surrogate_c1"
EXTRA_FEATS = ROOT / "features/output/external/surrogate_extra_features_v1.0.parquet"
EXTRA_MONET = ROOT / "concepts/output/monet_concepts_v1__surrogate_extra.parquet"
EXTRA_PREDS = OUT / "extra_predictions/predictions.csv"
OOF_GLOB = str(ROOT / "uncertaintyNet-main/output/narval_stageb_kdv8f_oof_k*_s3_*/run/predictions.csv")


def fail_without_sam(area_ratio, mean_prob):
    """quality_flag_v1's 'fail' test without the SAM-agreement term (no SAM masks for the extra images)."""
    return (area_ratio < 0.002) | (area_ratio > 0.85) | (mean_prob < 0.80)


def base_table():
    table = pd.read_parquet(FEATURES)
    preds = pd.read_csv(RUN)[["isic_id", "pred"]].rename(columns={"pred": "resnet_pred"})
    df = table.merge(preds, on="isic_id")
    df["label"] = df.label.astype(int)
    return df[df.extraction_error.fillna("") == ""].reset_index(drop=True)


def extra_rows(feats):
    x = pd.read_parquet(EXTRA_FEATS).rename(columns={"image_id": "isic_id"})
    m = pd.read_parquet(EXTRA_MONET)
    m = m[["isic_id", *[src for src, _ in MONET_FEATURES.values()]]].rename(
        columns={src: fid for fid, (src, _) in MONET_FEATURES.items()})
    p = pd.read_csv(EXTRA_PREDS)[["isic_id", "pred"]].rename(columns={"pred": "resnet_pred"})
    x = x.merge(m, on="isic_id").merge(p, on="isic_id")
    x = x[(x.extraction_error.fillna("") == "") & ~fail_without_sam(x.unet_area_ratio, x.unet_mean_prob_in_mask)]
    ova = np.where(x.diagnosis.str.startswith("Melanoma"), "Melanoma", "other")
    return x.assign(split="train", mask_quality="extra", ova_class=ova)[["isic_id", "split", "label", "ova_class",
                                                                         "mask_quality", "resnet_pred", *feats]]


def oof_calls():
    files = sorted(glob.glob(OOF_GLOB))
    assert len(files) == 5, f"expected 5 fold runs, found {len(files)}"
    p = pd.concat([pd.read_csv(f) for f in files])
    p = p[p.split == "oof"]
    assert p.isic_id.is_unique
    return dict(zip(p.isic_id, p.pred))


def main():
    df = base_table()
    feats = feature_columns(df)
    oof = oof_calls()
    ex = extra_rows(feats)
    tr = df[(df.split == "train") & (df.mask_quality != "fail")]
    va_fit = df[(df.split == "val") & (df.mask_quality != "fail")]
    val, test = df[df.split == "val"], df[df.split == "test"]
    tr_oof = tr.assign(resnet_pred=tr.isic_id.map(oof))
    assert tr_oof.resnet_pred.notna().all(), "a split-v2 train row has no out-of-fold call"
    res = {"n_extra_rows": len(ex), "n_extra_melanoma": int((ex.ova_class == "Melanoma").sum()),
           "extra_resnet_malignant_rate_on_melanoma": float(ex[ex.ova_class == "Melanoma"].resnet_pred.mean()),
           "train_melanoma_malignant_rate": {"in_sample": float(tr[tr.ova_class == "Melanoma"].resnet_pred.mean()),
                                             "out_of_fold": float(tr_oof[tr_oof.ova_class == "Melanoma"].resnet_pred.mean())},
           "train_calls_changed_by_oof": float((tr_oof.resnet_pred != tr.resnet_pred).mean()), "arms": {}}
    arms = {"current": tr, "+E": pd.concat([tr, ex]), "O": tr_oof, "O+E": pd.concat([tr_oof, ex])}
    calls = {}
    for name, t in arms.items():
        m, _, _ = fit_tuned(t[feats], t.resnet_pred.astype(int), va_fit[feats], va_fit.resnet_pred)
        calls[name] = lambda d, m=m: (m.predict_proba(d[feats])[:, 1] >= 0.5).astype(int)
        res["arms"][name] = {"n_train": len(t), "val": scores(val, calls[name])}
        print(name, {k: round(v, 4) for k, v in res["arms"][name]["val"].items() if isinstance(v, float)}, flush=True)
    b = res["arms"]["current"]["val"]
    best = max(res["arms"], key=lambda k: (res["arms"][k]["val"]["min_class"], res["arms"][k]["val"]["overall"]))
    v = res["arms"][best]["val"]
    t_cur, t_best = scores(test, calls["current"]), scores(test, calls[best])
    checks = {"val_score>=current+0.01": v["min_class"] >= b["min_class"] + 0.01,
              "val_overall>=current-0.005": v["overall"] >= b["overall"] - 0.005,
              "test_overall>=current-0.005": t_best["overall"] >= t_cur["overall"] - 0.005,
              "test_benign>=current-0.005": t_best["benign"] >= t_cur["benign"] - 0.005}
    res["selection"] = {"best": best, "checks": checks, "adopt": best != "current" and all(checks.values()),
                        "test": {"current": t_cur, best: t_best}}
    (OUT / "report_c1b.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["selection"], indent=1))


if __name__ == "__main__":
    main()
