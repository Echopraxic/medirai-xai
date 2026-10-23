"""
Multi-photo lesion averaging (decision 2026-10-22; idea from Liu et al. 2020, which combines several photos per case).
When a lesion has several photos (MIDAS: 6 in and 1 ft; some ISIC lesions), every photo gets the lesion's aggregated
probability. Rule fixed before looking: mean of calibrated probabilities (primary), max (secondary).

Selection data only (split v2 val + MIDAS dev_val), using the final model's calibrated TTA probabilities from
eval/final_model_report.py (scores.npz). Image-level metrics keep the same denominator as the per-photo baseline;
95% CIs from a paired bootstrap over lesions.

    python eval/lesion_aggregation.py --scores eval/output/stage_c/final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def selection_frames():
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    val = v2[v2.split == "val"]
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone") & (mid.split == "dev_val")]
    return {"v2_val": pd.DataFrame({"label": val.label.astype(int).to_numpy(),
                                    "lesion": val.lesion_id.fillna(val.isic_id).to_numpy(),
                                    "melanoma": val.diagnosis_3.fillna("").str.contains("Melanoma").to_numpy()}),
            "midas_dev_val": pd.DataFrame({"label": mid.label.astype(int).to_numpy(), "lesion": mid.lesion_id.to_numpy(),
                                           "melanoma": mid.midas_path.str.contains("melanoma").to_numpy()})}


def rates(y, p, t, mel):
    pred = p >= t
    return {"sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
            "melanoma_recall": float(pred[mel].mean()) if mel.any() else None}


def paired_auc_ci(d, a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    les = d.lesion.unique()
    idx = {l: np.flatnonzero(d.lesion.to_numpy() == l) for l in les}
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([idx[l] for l in rng.choice(les, len(les))])
        y = d.label.to_numpy()[take]
        if len(set(y)) == 2:
            diffs.append(roc_auc_score(y, d[b].to_numpy()[take]) - roc_auc_score(y, d[a].to_numpy()[take]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main(opts):
    s = np.load(ROOT / opts.scores / "scores.npz")
    t = json.loads((ROOT / opts.scores / "final_model_report.json").read_text())["threshold_standard"]
    res = {"threshold": t}
    for name, d in selection_frames().items():
        d = d.assign(photo=s[f"{name}_prob"])
        g = d.groupby("lesion").photo
        d["mean"], d["max"] = g.transform("mean"), g.transform("max")
        r = {"n_images": len(d), "n_lesions": int(d.lesion.nunique()),
             "multi_photo_lesions": int((d.lesion.value_counts() > 1).sum())}
        for col in ("photo", "mean", "max"):
            r[col] = {"auroc": float(roc_auc_score(d.label, d[col])),
                      **rates(d.label.to_numpy(), d[col].to_numpy(), t, d.melanoma.to_numpy())}
        for col in ("mean", "max"):
            r[f"{col}_minus_photo_auroc_ci95"] = paired_auc_ci(d, "photo", col)
        res[name] = r
    out = ROOT / "eval/output/stage_c/lesion_aggregation.json"
    out.write_text(json.dumps(res, indent=1))
    for name in ("v2_val", "midas_dev_val"):
        r = res[name]
        print(name, f"lesions {r['n_lesions']} (multi-photo {r['multi_photo_lesions']})")
        for col in ("photo", "mean", "max"):
            x = r[col]
            print(f"  {col:5s} AUROC {x['auroc']:.3f}  sens {x['sensitivity']:.3f}  spec {x['specificity']:.3f}  "
                  f"melanoma {x['melanoma_recall'] if x['melanoma_recall'] is None else round(x['melanoma_recall'], 3)}")
        print("  mean-photo CI", [round(v, 3) for v in r["mean_minus_photo_auroc_ci95"]],
              "| max-photo CI", [round(v, 3) for v in r["max_minus_photo_auroc_ci95"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="eval/output/stage_c/final")
    main(ap.parse_args())
