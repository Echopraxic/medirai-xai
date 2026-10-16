"""
Cross-check S1's malignant-vs-benign tree drivers against S2's one-vs-all drivers (plan T2.7), and explain S1's
low melanoma recall.

All models were trained on the same feature table with the same tuning (trees/train_trees.py fit_tuned), so their
TreeSHAP values (log-odds, tree_path_dependent) are on comparable scales. Everything is computed on the test split.

  1. Drivers per model: mean |SHAP| per feature and per concept group (SHAP summed within the group first, as in
     the top-3 selector), plus each feature's direction (corr(value, SHAP): + means higher values push toward the
     model's positive class).
  2. Agreement: Spearman correlation of per-feature mean |SHAP| between S1's binary tree and every other model;
     overlap of the top-10 features.
  3. Melanoma: what S1's tree does with melanomas. Recall, mean grouped SHAP on melanomas (which concepts push
     them toward benign), how close melanomas sit to nevi vs BCC on the binary tree's top drivers, and whether the
     binary score behaves like a BCC detector (correlation with the BCC one-vs-all score).

    python trees/compare_drivers.py --features features/output/feature_table_v1.0.parquet \
        --s1 trees/output/s1_binary_v1.0 --s2 trees/output_v1.0 --out trees/output_v1.0/driver_comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

from train_trees import OVA_CLASSES, feature_columns

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"A": "asymmetry", "B": "border", "C": "color", "T": "texture/surface", "S": "shape"}


def load_model(path):
    m = xgb.XGBClassifier()
    m.load_model(path)
    return m


def shap_frame(model, X):
    sv = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X)
    return pd.DataFrame(sv, columns=X.columns, index=X.index)


def drivers(sv, X):
    feat = pd.DataFrame({
        "mean_abs_shap": sv.abs().mean(),
        "direction": [np.corrcoef(X[f], sv[f])[0, 1] if X[f].std() > 0 and sv[f].std() > 0 else np.nan
                      for f in X.columns]})
    grouped = pd.DataFrame({g: sv[[f for f in X.columns if f[0] == g]].sum(1) for g in GROUPS if any(f[0] == g for f in X.columns)})
    group = grouped.abs().mean().rename(index=GROUPS)
    return feat, group, grouped.rename(columns=GROUPS)


def main(opts):
    df = pd.read_parquet(opts.features)
    feats = feature_columns(df)
    test = df[(df.split == "test") & df[feats].notna().any(axis=1)].set_index("isic_id")
    X = test[feats]
    models = {"S1_binary": Path(opts.s1) / "binary_gt.json", "S2_binary_gt": Path(opts.s2) / "binary_gt.json",
              "S2_surrogate": Path(opts.s2) / "binary_surrogate.json"}
    models.update({f"ova_{c}": Path(opts.s2) / f"ova_{c}.json" for c in OVA_CLASSES})

    feat_imp, group_imp, grouped, shap_by = {}, {}, {}, {}
    for name, path in models.items():
        sv = shap_frame(load_model(path), X)
        shap_by[name] = sv
        feat_imp[name], group_imp[name], grouped[name] = drivers(sv, X)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    groups_tbl = pd.DataFrame(group_imp).T.round(3)
    groups_share = groups_tbl.div(groups_tbl.sum(1), axis=0).round(3)
    ref = feat_imp["S1_binary"].mean_abs_shap
    top10_ref = set(ref.nlargest(10).index)
    agreement = {name: {"spearman_mean_abs_shap": float(spearmanr(ref, f.mean_abs_shap).statistic),
                        "top10_overlap": len(top10_ref & set(f.mean_abs_shap.nlargest(10).index))}
                 for name, f in feat_imp.items() if name != "S1_binary"}

    # ---- melanoma
    preds = pd.read_csv(Path(opts.s1) / "tree_predictions.csv").set_index("isic_id").reindex(test.index)
    s2p = pd.read_csv(Path(opts.s2) / "tree_predictions.csv").set_index("isic_id").reindex(test.index)
    cls = test.ova_class
    mel, nev, bcc = cls == "Melanoma", cls == "Nevus", cls == "BCC"
    g_s1 = grouped["S1_binary"]
    top_drivers = ref.nlargest(5).index.tolist()
    medians = test.loc[mel | nev | bcc, top_drivers].groupby(cls[mel | nev | bcc]).median().round(3)
    # distance of melanoma medians to nevus vs BCC medians on the top drivers, in units of the feature's test IQR
    iqr = (X[top_drivers].quantile(0.75) - X[top_drivers].quantile(0.25)).replace(0, np.nan)
    closer_to = {f: ("Nevus" if abs(medians.loc["Melanoma", f] - medians.loc["Nevus", f]) <
                     abs(medians.loc["Melanoma", f] - medians.loc["BCC", f]) else "BCC") for f in top_drivers}
    gap_nevus = ((medians.loc["Melanoma"] - medians.loc["Nevus"]).abs() / iqr).round(2).to_dict()
    gap_bcc = ((medians.loc["Melanoma"] - medians.loc["BCC"]).abs() / iqr).round(2).to_dict()
    malignant = test.label == 1
    melanoma = {
        "n_test": int(mel.sum()),
        "S1_binary_recall": float((preds.loc[mel, "pred"] == 1).mean()),
        "S1_binary_recall_by_class": {c: float((preds.loc[cls == c, "pred"] == 1).mean())
                                      for c in ("BCC", "SCC", "Melanoma") if (cls == c).any()},
        "S2_ova_as_binary_recall": float((s2p.loc[mel, "ova_pred"] == 1).mean()),
        "S2_ova_melanoma_top_class_rate": float((s2p.loc[mel, "ova_pred_class"] == "Melanoma").mean()),
        "resnet_recall": float((s2p.loc[mel, "resnet_pred"] == 1).mean()),
        "S1_mean_grouped_shap_on_melanoma": g_s1[mel].mean().round(3).to_dict(),
        "S1_mean_grouped_shap_on_bcc": g_s1[bcc].mean().round(3).to_dict(),
        "S1_mean_grouped_shap_on_nevus": g_s1[nev].mean().round(3).to_dict(),
        "top_driver_medians": medians.to_dict(),
        "melanoma_closer_to": closer_to,
        "melanoma_gap_to_nevus_iqr": gap_nevus,
        "melanoma_gap_to_bcc_iqr": gap_bcc,
        "corr_S1_p_vs_ova_BCC_score_malignant": float(spearmanr(preds.loc[malignant, "p_malignant"],
                                                                s2p.loc[malignant, "ova_score_BCC"]).statistic),
        "corr_S1_p_vs_ova_Melanoma_score_malignant": float(spearmanr(preds.loc[malignant, "p_malignant"],
                                                                     s2p.loc[malignant, "ova_score_Melanoma"]).statistic),
    }

    report = {"n_test": len(test), "group_mean_abs_shap": groups_tbl.to_dict("index"),
              "group_share": groups_share.to_dict("index"), "agreement_with_S1_binary": agreement,
              "S1_top10": ref.nlargest(10).round(3).to_dict(),
              "S1_top10_direction": feat_imp["S1_binary"].direction[ref.nlargest(10).index].round(2).to_dict(),
              "ova_Melanoma_top10": feat_imp["ova_Melanoma"].mean_abs_shap.nlargest(10).round(3).to_dict(),
              "ova_Melanoma_top10_direction": feat_imp["ova_Melanoma"].direction[
                  feat_imp["ova_Melanoma"].mean_abs_shap.nlargest(10).index].round(2).to_dict(),
              "melanoma": melanoma}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    pd.DataFrame({k: v.mean_abs_shap for k, v in feat_imp.items()}).round(4).to_csv(out / "feature_mean_abs_shap.csv")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "features/output/feature_table_v1.0.parquet"))
    ap.add_argument("--s1", default=str(ROOT / "trees/output/s1_binary_v1.0"))
    ap.add_argument("--s2", default=str(ROOT / "trees/output_v1.0"))
    ap.add_argument("--out", default=str(ROOT / "trees/output_v1.0/driver_comparison"))
    main(ap.parse_args())
