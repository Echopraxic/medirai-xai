"""
Which signals flag the final model's errors, especially on shifted data (K8: entropy AUROC for errors was 0.58 on the
locked sets, below 0.5 on DDI and DDI-2)? Selection data only (split v2 val + MIDAS dev_val phone photos).

Decision = lesion-averaged calibrated probability (eval/lesion_aggregation.py) at the standard threshold; error =
decision != label. Candidate signals (fixed before looking):
  entropy            binary entropy of the decision probability (current deferral signal)
  photo_disagreement std of the per-photo probabilities within the lesion (0 for single-photo lesions): a second look
  scope_distance     DINOv2 Mahalanobis distance to the training data (I7 detector)
  meta               L2 logistic error predictor on [entropy, margin to threshold, photo_disagreement, scope_distance,
                     12 MONET concepts], 5-fold CV grouped by patient, trained on both sets pooled
Reported per set: AUROC for errors, and accuracy retained at 30% deferral when deferring by each signal.

    python eval/deferral_signals.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from stacking import build  # noqa: E402


def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    d, cols = build("eval/output/stage_c/final")
    t = json.loads((ROOT / "eval/output/stage_c/final/final_model_report.json").read_text())["threshold_standard"]
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv").set_index("isic_id")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv").set_index("image_id")
    d["lesion"] = [v2.loc[i, "lesion_id"] if s == "v2_val" and pd.notna(v2.loc[i, "lesion_id"]) else
                   (i if s == "v2_val" else mid.loc[i, "lesion_id"]) for i, s in zip(d.id, d.set)]
    g = d.groupby("lesion").p
    d["p_dec"] = g.transform("mean")
    d["photo_disagreement"] = g.transform("std").fillna(0.0)
    d["entropy"] = entropy(d.p_dec)
    d["margin"] = -np.abs(d.p_dec - t)
    d["error"] = ((d.p_dec >= t).astype(int) != d.label).astype(int)
    feats = ["entropy", "margin", "photo_disagreement", "scope_distance"] + [c for c in cols if c.startswith("M_")]
    meta = np.zeros(len(d))
    strat = d.error.astype(str) + "_" + d.set
    for rep in range(3):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(d, strat, d.group):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
            m.fit(d.iloc[tr][feats], d.error.iloc[tr])
            meta[te] += m.predict_proba(d.iloc[te][feats])[:, 1] / 3
    d["meta"] = meta
    res = {}
    for s in ("v2_val", "midas_dev_val"):
        x = d[d.set == s]
        r = {"n": len(x), "error_rate": float(x.error.mean())}
        for sig in ("entropy", "photo_disagreement", "scope_distance", "meta"):
            keep = x[sig] <= x[sig].quantile(0.7)
            r[sig] = {"auroc_for_errors": float(roc_auc_score(x.error, x[sig])),
                      "accuracy_at_30pct_deferral": float(1 - x.error[keep].mean())}
        res[s] = r
        print(s, f"n {r['n']} error rate {r['error_rate']:.3f}")
        for sig in ("entropy", "photo_disagreement", "scope_distance", "meta"):
            print(f"  {sig:20s} AUROC for errors {r[sig]['auroc_for_errors']:.3f}   accuracy at 30% deferral {r[sig]['accuracy_at_30pct_deferral']:.3f}")
    (ROOT / "eval/output/stage_c/deferral_signals.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
