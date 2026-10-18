"""
Surrogate-fidelity sweep (criterion C1): which training choices make the feature tree agree more with ResNet50?

Variants (each with the train_trees grid, early stopping on val):
  hard_balanced  current train_trees surrogate: ResNet hard labels, scale_pos_weight, threshold 0.5
  hard           hard labels, no class reweighting
  soft           distillation: regression on ResNet's p_malignant (reg:logistic)
  *_allmasks     also train on rows whose mask_quality is 'fail' (still never a feature)
  *_thr          decision threshold chosen on val to maximise per-class-mean agreement

Selection uses validation only: mean of the agreement on ResNet-benign and ResNet-malignant calls (balanced
fidelity), then overall agreement. Test numbers are reported for every variant but never used to choose.
Writes a report only; trees/output_* stay as they are.

    python trees/surrogate_sweep.py --features features/output/feature_table_v1.0.parquet \
        --predictions uncertaintyNet-main/output/rorqual_v2_lr3e-4_20261014/resnet50_det/run/predictions.csv \
        --out trees/output_v1.0_detv2/surrogate_sweep
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from train_trees import SEED, feature_columns

GRID = {"max_depth": [3, 4, 6, 8], "min_child_weight": [1, 5], "subsample": [0.8], "colsample_bytree": [0.8]}


def fidelity(pred, resnet_pred, label=None):
    pred, resnet_pred = np.asarray(pred), np.asarray(resnet_pred)
    per = {int(c): float((pred[resnet_pred == c] == c).mean()) for c in (0, 1)}
    out = {"overall": float((pred == resnet_pred).mean()), "on_resnet_benign": per[0], "on_resnet_malignant": per[1],
           "balanced": (per[0] + per[1]) / 2}
    if label is not None:
        label = np.asarray(label)
        out["by_true_label"] = {int(c): float((pred[label == c] == resnet_pred[label == c]).mean()) for c in (0, 1)}
        wrong = resnet_pred != label
        out["on_resnet_wrong"] = float((pred[wrong] == resnet_pred[wrong]).mean())
    return out


def fit(X_tr, y_tr, X_va, y_va, soft, balanced):
    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        kw = dict(n_estimators=1500, learning_rate=0.03, early_stopping_rounds=50, tree_method="hist",
                  random_state=SEED, n_jobs=6, **params)
        if soft:
            m = xgb.XGBRegressor(objective="reg:logistic", eval_metric="logloss", **kw)
        else:
            pos = y_tr.mean()
            m = xgb.XGBClassifier(eval_metric="logloss", scale_pos_weight=(1 - pos) / pos if balanced else 1.0, **kw)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (m.best_score, m, {**params, "best_iteration": int(m.best_iteration)})
    return best[1], best[2]


def proba(m, X):
    return m.predict(X) if isinstance(m, xgb.XGBRegressor) else m.predict_proba(X)[:, 1]


def best_threshold(p, resnet_pred):
    grid = np.round(np.arange(0.2, 0.81, 0.01), 2)
    scores = [(fidelity(p >= t, resnet_pred)["balanced"], fidelity(p >= t, resnet_pred)["overall"], t) for t in grid]
    return float(max(scores)[2])


def main(opts):
    df = pd.read_parquet(opts.features)
    preds = pd.read_csv(opts.predictions)[["isic_id", "pred", "p_malignant"]].rename(
        columns={"pred": "resnet_pred", "p_malignant": "resnet_p"})
    df = df.merge(preds, on="isic_id", how="inner")
    df["label"] = df.label.astype(int)
    feats = feature_columns(df)
    usable = df.extraction_error.fillna("") == ""
    va = df[usable & (df.split == "val")]
    te = df[df.split == "test"]
    rows = []
    for target, balanced, allmasks in [("hard", True, False), ("hard", False, False), ("soft", False, False),
                                       ("hard", False, True), ("soft", False, True)]:
        trm = usable & (df.split == "train") & (allmasks | (df.mask_quality != "fail"))
        tr = df[trm]
        vam = va if allmasks else va[va.mask_quality != "fail"]
        soft = target == "soft"
        y_tr, y_va = (tr.resnet_p, vam.resnet_p) if soft else (tr.resnet_pred, vam.resnet_pred)
        m, params = fit(tr[feats], y_tr, vam[feats], y_va, soft, balanced)
        name = f"{target}{'_balanced' if balanced else ''}{'_allmasks' if allmasks else ''}"
        p_va, p_te = proba(m, va[feats]), proba(m, te[feats])
        for thr_name, thr in (("0.5", 0.5), ("val", best_threshold(p_va, va.resnet_pred.values))):
            rows.append({"variant": name + ("" if thr_name == "0.5" else "_thr"), "threshold": thr, "params": params,
                         "n_train": len(tr), "val": fidelity(p_va >= thr, va.resnet_pred),
                         "test": fidelity(p_te >= thr, te.resnet_pred, te.label)})
            print(f"{rows[-1]['variant']:28s} thr {thr:.2f}  val bal {rows[-1]['val']['balanced']:.3f} "
                  f"overall {rows[-1]['val']['overall']:.3f} | test overall {rows[-1]['test']['overall']:.3f} "
                  f"benign-call {rows[-1]['test']['on_resnet_benign']:.3f} malig-call {rows[-1]['test']['on_resnet_malignant']:.3f}",
                  flush=True)
    chosen = max(rows, key=lambda r: (r["val"]["balanced"], r["val"]["overall"]))
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"selection": "val balanced fidelity, then val overall", "chosen": chosen["variant"],
              "baseline": "hard_balanced", "variants": rows,
              "predictions": str(opts.predictions), "features": str(opts.features)}
    (out / "surrogate_sweep.json").write_text(json.dumps(report, indent=1))
    print("chosen:", chosen["variant"], json.dumps(chosen["test"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/output/feature_table_v1.0.parquet")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
