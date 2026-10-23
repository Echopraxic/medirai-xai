"""
Second-stage stacking (decision 2026-10-22; pattern of the ISIC 2024 top solutions: a tabular model on image-model
outputs plus extra features). Fixed before looking:

  features  final model's calibrated logit, p(melanoma) head, 12 named MONET concept scores, DINOv2 scope distance
            (Mahalanobis, fit on split v2 train) -- available for internal and MIDAS photos alike
  models    L2 logistic regression (C = 1, standardized) and XGBoost (depth 2, 200 trees, lr 0.05)
  data      selection data only (split v2 val + MIDAS dev_val phone photos); the image model's train-split outputs are
            overfit, so they are never used. 5-fold CV grouped by patient/lesion, stratified on label, 3 repeats
  report    out-of-fold AUROC per set vs the image model alone on the same images, paired bootstrap 95% CI

    python eval/stacking.py --run narval_stageb_kdv6_mel_s1_20261022/run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "eval/output/interpretability/rorqual_v2_lr3e-4_20261014/artifacts/scope_model.npz"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mahalanobis(emb, s):
    z = (emb - s["pca_mean"]) @ s["pca_components"].T - s["location"]
    return np.einsum("ij,jk,ik->i", z, s["precision"], z)


def build(scores_dir):
    s = np.load(ROOT / scores_dir / "scores.npz")
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    val = v2[v2.split == "val"]
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone") & (mid.split == "dev_val")]
    frames = [pd.DataFrame({"id": val.isic_id.to_numpy(), "set": "v2_val", "label": val.label.astype(int).to_numpy(),
                            "group": val.group_id.to_numpy(), "p": s["v2_val_prob"], "pmel": s["v2_val_pmel"]}),
              pd.DataFrame({"id": mid.image_id.to_numpy(), "set": "midas_dev_val", "label": mid.label.astype(int).to_numpy(),
                            "group": mid.patient_id.to_numpy(), "p": s["midas_dev_val_prob"], "pmel": s["midas_dev_val_pmel"]})]
    d = pd.concat(frames, ignore_index=True)
    monet = pd.concat([pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1.parquet"),
                       pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1__midas.parquet")]).drop_duplicates("isic_id")
    mcols = [c for c in monet.columns if c.startswith("M_") and not c.startswith("M_artifact_")]
    d = d.merge(monet.rename(columns={"isic_id": "id"})[["id"] + mcols], on="id", how="left")
    emb = []
    for f in ("dinov2.npz", "dinov2__midas.npz"):
        z = np.load(ROOT / "classifier/output/embeddings" / f, allow_pickle=True)
        emb.append(pd.DataFrame(z["emb"].astype(np.float64), index=z["isic_id"]))
    emb = pd.concat(emb)
    emb = emb[~emb.index.duplicated()]
    d["scope_distance"] = mahalanobis(emb.loc[d.id].to_numpy(), np.load(SCOPE))
    d["logit"] = logit(d.p)
    return d, ["logit", "pmel", "scope_distance"] + mcols


def oof(d, cols, model, seed):
    pred = np.zeros(len(d))
    strat = d.label.astype(str) + "_" + d.set
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d, strat, d.group):
        m = model()
        m.fit(d.iloc[tr][cols], d.label.iloc[tr])
        pred[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
    return pred


def paired_ci(y, a, b, groups, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in g}
    diffs = []
    for _ in range(n_boot):
        t = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if len(set(y[t])) == 2:
            diffs.append(roc_auc_score(y[t], b[t]) - roc_auc_score(y[t], a[t]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main(opts):
    d, cols = build(opts.scores)
    models = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
              "xgboost": lambda: xgb.XGBClassifier(max_depth=2, n_estimators=200, learning_rate=0.05, subsample=0.8,
                                                   colsample_bytree=0.8, eval_metric="logloss", random_state=0)}
    res = {"features": cols}
    for name, model in models.items():
        p = np.mean([oof(d, cols, model, seed) for seed in range(3)], axis=0)
        res[name] = {}
        for s in ("v2_val", "midas_dev_val"):
            m = (d.set == s).to_numpy()
            y = d.label.to_numpy()[m]
            base, stack = d.p.to_numpy()[m], p[m]
            res[name][s] = {"image_model_auroc": float(roc_auc_score(y, base)), "stacked_auroc": float(roc_auc_score(y, stack)),
                            "diff_ci95": paired_ci(y, base, stack, d.group.to_numpy()[m])}
        print(name, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else [round(x, 3) for x in vv])
                                    for kk, vv in v.items()} for k, v in res[name].items()}))
    out = ROOT / "eval/output/stage_c/stacking.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="eval/output/stage_c/final")
    main(ap.parse_args())
