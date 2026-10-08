"""
S1's binary tree: malignant vs benign on the shared feature table, ground-truth labels, + TreeSHAP (plan S1-5).

Uses the same tuning as trees/train_trees.py (XGBoost, small grid on validation log-loss, early stopping)
so its drivers are directly comparable with S2's one-vs-all trees (T2.7). Needs no ResNet predictions.
Rows whose feature extraction failed (all features NaN) are left out of training and evaluation.

Outputs (in --out):
  binary_gt.json           the XGBoost model
  tree_predictions.csv     isic_id, split, label, ova_class, source, p_malignant, pred
  shap_values.parquet      isic_id + one TreeSHAP column per feature (log-odds of malignant) + base_value
  importance.csv           mean |SHAP| on test, per feature and per concept group, with sign vs label
  report.json              val/test performance overall, per source and per one-vs-all class

    python trees/s1_binary_tree.py --features features/output/feature_table_v0.2.parquet \
        --out trees/output/s1_binary_v0.2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from train_trees import feature_columns, fit_tuned

ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {"A": "asymmetry", "B": "border", "C": "color", "D": "diameter", "T": "texture/surface",
               "S": "shape"}


def performance(y, p, pred):
    y = np.asarray(y)
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean()),
           "sensitivity": float(pred[y == 1].mean()) if (y == 1).any() else float("nan"),
           "specificity": float(1 - pred[y == 0].mean()) if (y == 0).any() else float("nan")}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    return out


def main(opts):
    table = pd.read_parquet(opts.features)
    split = pd.read_csv(opts.split, usecols=["isic_id", "source"])
    df = table.merge(split, on="isic_id", how="left")
    df["label"] = df.label.astype(int)
    feats = feature_columns(df)
    usable = df[feats].notna().any(axis=1)
    tr, va = df[usable & (df.split == "train")], df[usable & (df.split == "val")]

    model, params, val_auc = fit_tuned(tr[feats], tr.label, va[feats], va.label)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "binary_gt.json")

    d = df[usable].reset_index(drop=True)
    d["p_malignant"] = model.predict_proba(d[feats])[:, 1]
    d["pred"] = (d.p_malignant >= 0.5).astype(int)
    d[["isic_id", "split", "label", "ova_class", "source", "p_malignant", "pred"]].to_csv(
        out / "tree_predictions.csv", index=False)

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = pd.DataFrame(explainer.shap_values(d[feats]), columns=feats)
    sv.insert(0, "isic_id", d.isic_id)
    sv["base_value"] = float(np.ravel(explainer.expected_value)[0])
    sv.to_parquet(out / "shap_values.parquet", index=False)

    # global drivers on test: mean |SHAP|; "direction" = correlation of feature value with its SHAP value
    # (+ means higher values push toward malignant)
    te = d.split == "test"
    rows = []
    for f in feats:
        x, s = d.loc[te, f], sv.loc[te, f]
        ok = x.notna() & (x.std() > 0)
        rows.append({"level": "feature", "name": f, "group": GROUP_NAMES[f[0]], "mean_abs_shap": float(s.abs().mean()),
                     "direction": float(np.corrcoef(x[ok], s[ok])[0, 1]) if ok.sum() > 2 else float("nan")})
    for g, name in GROUP_NAMES.items():
        cols = [f for f in feats if f[0] == g]
        if cols:
            rows.append({"level": "group", "name": name, "group": name,
                         "mean_abs_shap": float(sv.loc[te, cols].sum(1).abs().mean()), "direction": float("nan")})
    imp = pd.DataFrame(rows).sort_values(["level", "mean_abs_shap"], ascending=[True, False])
    imp.to_csv(out / "importance.csv", index=False)

    test = d[te]
    report = {
        "features": opts.features, "n_features": len(feats),
        "n_train": len(tr), "n_val": len(va), "n_test": int(te.sum()),
        "n_excluded_extraction_error": int((~usable).sum()),
        "params": params, "val_auroc": val_auc,
        "test": performance(test.label, test.p_malignant, test.pred),
        "test_by_source": {k: performance(g.label, g.p_malignant, g.pred)
                           for k, g in test.groupby("source") if len(g) >= 20},
        "test_by_ova_class": {k: performance(g.label, g.p_malignant, g.pred) for k, g in test.groupby("ova_class")},
        "top_features": imp[imp.level == "feature"].head(10)[["name", "mean_abs_shap", "direction"]].to_dict("records"),
        "groups": imp[imp.level == "group"][["name", "mean_abs_shap"]].to_dict("records"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("val_auroc", "test", "test_by_source", "groups", "top_features")},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "features/output/feature_table_v0.2.parquet"))
    ap.add_argument("--split", default=str(ROOT / "splits/isic_clinical_v2.csv"))
    ap.add_argument("--out", default=str(ROOT / "trees/output/s1_binary_v0.2"))
    main(ap.parse_args())
