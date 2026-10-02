"""
Tree models on the v0 feature table + surrogate fidelity against ResNet50 (plan T2.2, T2.5; criterion C1).

Models (XGBoost, hyper-parameters chosen on the validation split only):
  binary_gt         malignant vs benign, trained on ground truth (S1's model family; here for comparison)
  binary_surrogate  trained on ResNet50's predicted labels -> a true surrogate of the CNN
  ova_<class>       one-vs-all per diagnosis class (BCC, Nevus, SCC, SK, Melanoma, Other), ground truth

Fidelity = agreement between a tree's malignant/benign call and ResNet50's, on the test split, overall and
per true class, per ResNet class, per source, per mask quality, and on the cases ResNet50 gets wrong.

    python trees/train_trees.py --features features/output/feature_table_v0.1.parquet \
        --predictions uncertaintyNet-main/output/s2_local/resnet50_det/run_X/predictions.csv --out trees/output
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

OVA_CLASSES = ["BCC", "Nevus", "SCC", "SK", "Melanoma", "Other"]
MALIGNANT_CLASSES = {"BCC", "SCC", "Melanoma"}
GRID = {"max_depth": [3, 4, 6], "min_child_weight": [1, 5], "subsample": [0.8], "colsample_bytree": [0.8]}
SEED = 42


def feature_columns(df):
    return [c for c in df.columns if len(c) > 2 and c[1] == "_" and c[0] in "ABCDTS"]


def fit_tuned(X_tr, y_tr, X_va, y_va, seed=SEED):
    """Small grid on validation log-loss with early stopping; returns (model, params, val_auc)."""
    pos = y_tr.mean()
    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, early_stopping_rounds=50,
                              scale_pos_weight=(1 - pos) / pos, eval_metric="logloss", tree_method="hist",
                              random_state=seed, n_jobs=6, **params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        loss = m.best_score
        if best is None or loss < best[0]:
            best = (loss, m, params)
    _, model, params = best
    p = model.predict_proba(X_va)[:, 1]
    return model, {**params, "best_iteration": int(model.best_iteration)}, float(roc_auc_score(y_va, p))


def agreement(a, b):
    return float((np.asarray(a) == np.asarray(b)).mean()) if len(a) else float("nan")


def fidelity_table(test, tree_col):
    out = {"overall": agreement(test[tree_col], test.resnet_pred), "n": len(test)}
    out["by_true_label"] = {int(k): agreement(g[tree_col], g.resnet_pred) for k, g in test.groupby("label")}
    out["by_resnet_pred"] = {int(k): agreement(g[tree_col], g.resnet_pred) for k, g in test.groupby("resnet_pred")}
    out["by_ova_class"] = {k: agreement(g[tree_col], g.resnet_pred) for k, g in test.groupby("ova_class")}
    out["by_source"] = {k: agreement(g[tree_col], g.resnet_pred) for k, g in test.groupby("source") if len(g) >= 20}
    out["by_mask_quality"] = {k: agreement(g[tree_col], g.resnet_pred) for k, g in test.groupby("mask_quality")}
    wrong = test[test.resnet_pred != test.label]
    out["on_resnet_wrong"] = {"n": len(wrong), "agreement": agreement(wrong[tree_col], wrong.resnet_pred)}
    out["tree_accuracy_vs_truth"] = agreement(test[tree_col], test.label)
    return out


def main(opts):
    table = pd.read_parquet(opts.features)
    preds = pd.read_csv(opts.predictions)[["isic_id", "pred", "p_malignant", "entropy_of_expected"]]
    preds = preds.rename(columns={"pred": "resnet_pred", "p_malignant": "resnet_p_malignant"})
    df = table.merge(preds, on="isic_id", how="inner")
    df["source"] = df.attribution.str.slice(0, 4).map({"MILK": "MILK", "Memo": "MSKCC", "Fede": "UFES"}).fillna("other")
    df["label"] = df.label.astype(int)
    feats = feature_columns(df)
    usable = df.extraction_error.fillna("") == ""
    train_mask = usable & (df.split == "train") & (df.mask_quality != "fail")
    val_mask = usable & (df.split == "val") & (df.mask_quality != "fail")
    tr, va = df[train_mask], df[val_mask]
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"n_features": len(feats), "n_train": len(tr), "n_val": len(va),
              "n_test": int((df.split == "test").sum()), "train_excludes": "mask_quality == fail or extraction error"}

    models = {}
    for name, target in (("binary_gt", "label"), ("binary_surrogate", "resnet_pred")):
        m, params, val_auc = fit_tuned(tr[feats], tr[target], va[feats], va[target])
        models[name] = m
        report[name] = {"params": params, "val_auc_vs_target": val_auc}
        m.save_model(out / f"{name}.json")
        df[f"{name}_p"] = m.predict_proba(df[feats])[:, 1]
        df[f"{name}_pred"] = (df[f"{name}_p"] >= 0.5).astype(int)

    # one-vs-all over diagnosis classes; malignant probability = sum over malignant classes (+ Other's share)
    other_malignant_rate = float(tr.loc[tr.ova_class == "Other", "label"].mean())
    scores = {}
    for c in OVA_CLASSES:
        y_tr, y_va = (tr.ova_class == c).astype(int), (va.ova_class == c).astype(int)
        m, params, val_auc = fit_tuned(tr[feats], y_tr, va[feats], y_va)
        models[f"ova_{c}"] = m
        m.save_model(out / f"ova_{c}.json")
        scores[c] = m.predict_proba(df[feats])[:, 1]
        report[f"ova_{c}"] = {"params": params, "val_auc": val_auc}
    S = np.column_stack([scores[c] for c in OVA_CLASSES])
    P = S / S.sum(1, keepdims=True)
    df["ova_pred_class"] = np.array(OVA_CLASSES)[P.argmax(1)]
    weights = np.array([1.0 if c in MALIGNANT_CLASSES else (other_malignant_rate if c == "Other" else 0.0)
                        for c in OVA_CLASSES])
    df["ova_p_malignant"] = P @ weights
    df["ova_pred"] = (df.ova_p_malignant >= 0.5).astype(int)
    for i, c in enumerate(OVA_CLASSES):
        df[f"ova_score_{c}"] = P[:, i]

    test = df[df.split == "test"]
    report["test_performance"] = {
        "resnet": {"accuracy": agreement(test.resnet_pred, test.label),
                   "auroc": float(roc_auc_score(test.label, test.resnet_p_malignant))},
        **{name: {"accuracy": agreement(test[f"{name}_pred"], test.label),
                  "balanced_accuracy": float(balanced_accuracy_score(test.label, test[f"{name}_pred"])),
                  "auroc": float(roc_auc_score(test.label, test[f"{name}_p"]))}
           for name in ("binary_gt", "binary_surrogate")},
        "ova_as_binary": {"accuracy": agreement(test.ova_pred, test.label),
                          "auroc": float(roc_auc_score(test.label, test.ova_p_malignant))},
        "ova_per_class": {c: {"n_pos": int((test.ova_class == c).sum()),
                              "auroc": float(roc_auc_score(test.ova_class == c, test[f"ova_score_{c}"])),
                              "f1": float(f1_score(test.ova_class == c, test.ova_pred_class == c))}
                          for c in OVA_CLASSES},
        "ova_macro_f1": float(f1_score(test.ova_class, test.ova_pred_class, average="macro")),
    }
    report["fidelity"] = {name: fidelity_table(test, f"{name}_pred")
                          for name in ("binary_gt", "binary_surrogate", "ova")}
    # does the feature set encode the image source? Compare MSKCC benign lesions with other benign lesions
    # only: MSKCC is 99.5% benign, so on all lesions "predict MSKCC" would just pick up the benign signal.
    benign = df.label == 0
    src = (df.source == "MSKCC").astype(int)
    m_src, _, _ = fit_tuned(tr.loc[benign[train_mask], feats], src[train_mask & benign],
                            va.loc[benign[val_mask], feats], src[val_mask & benign])
    test_b = test[test.label == 0]
    report["source_shortcut_check"] = {
        "auroc_MSKCC_vs_other_benign_test": float(roc_auc_score(test_b.source == "MSKCC",
                                                                 m_src.predict_proba(test_b[feats])[:, 1])),
        "n_test_benign": len(test_b),
        "note": "benign-only; high AUROC = features identify the MSKCC source (99.5% benign) beyond lesion type"}

    keep = ["isic_id", "split", "label", "ova_class", "source", "mask_quality", "resnet_pred", "resnet_p_malignant",
            "entropy_of_expected", "binary_gt_p", "binary_gt_pred", "binary_surrogate_p", "binary_surrogate_pred",
            "ova_pred_class", "ova_p_malignant", "ova_pred"] + [f"ova_score_{c}" for c in OVA_CLASSES]
    df[keep].to_csv(out / "tree_predictions.csv", index=False)
    (out / "tree_report.json").write_text(json.dumps(report, indent=1))
    (out / "features_used.json").write_text(json.dumps(feats))
    print(json.dumps({k: report[k] for k in ("test_performance", "fidelity", "source_shortcut_check")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
