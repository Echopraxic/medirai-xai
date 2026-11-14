"""
Descriptive check of the adopted darker-skin scope detector (ToneScope, tone_model_scin_domain.npz, adopted 2026-10-30)
on PASSION: field photos from four African countries that played no part in fitting it (plan decision 2026-11-04).
Nothing is fitted or chosen here. Reported per split and overall:
  - flagged share by Fitzpatrick type (III, IV, V, VI); sensitivity to dark skin (V-VI) against the 0.85 design target;
  - specificity on III-IV; detector-score AUROC (V-VI vs III-IV); share routed to the phone threshold.
The plan's next step (TODO 2026-11-04): only if test-split sensitivity < 0.85 is a refit rule pre-registered.

    python eval/passion_tone_check.py   -> eval/output/passion/tone_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from interpret.skin_tone import ToneScope, _logistic

ROOT = Path(__file__).resolve().parents[1]


def summarize(d):
    pos, neg = d[d.dark == 1], d[d.dark == 0]
    return {"n_images": len(d), "n_subjects": int(d.subject_id.nunique()),
            "flagged_by_fst": {int(k): float(v) for k, v in d.groupby("fitzpatrick").flag.mean().items()},
            "n_by_fst": {int(k): int(v) for k, v in d.fitzpatrick.value_counts().sort_index().items()},
            "sensitivity_dark": float(pos.flag.mean()), "specificity_fst3_4": float(1 - neg.flag.mean()),
            "auroc": float(roc_auc_score(d.dark, d.p)) if d.dark.nunique() == 2 else None,
            "routed_phone": float(d.phone.mean()),
            "subject_level_sensitivity_dark": float(pos.groupby("subject_id").flag.max().mean())}


def main():
    lab = pd.read_csv(ROOT / "splits/passion_v1.csv", dtype={"image_id": str, "subject_id": str})
    z = np.load(ROOT / "classifier/output/embeddings/dinov2__passion.npz", allow_pickle=True)
    emb = dict(zip(z["isic_id"].astype(str), z["emb"].astype(np.float32)))
    ts = ToneScope()
    m = ts.m
    rows = []
    for i in lab.image_id:
        flag, p = ts(emb[i])
        ph = _logistic(emb[i], m["router_mean"], m["router_scale"], m["router_coef"], m["router_intercept"]) >= 0.5
        rows.append((bool(flag), p, bool(ph)))
    lab[["flag", "p", "phone"]] = pd.DataFrame(rows, index=lab.index)
    res = {"detector": "tone_model_scin_domain.npz (adopted 2026-10-30)", "design_sensitivity": 0.85,
           "all": summarize(lab), **{s: summarize(g) for s, g in lab.groupby("split")}}
    out = ROOT / "eval/output/passion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tone_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
