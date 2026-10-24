"""
S1-9 TreeSHAP on the final candidate's errors, internal test split (descriptive; nothing is selected).

Joins the final ResNet's v2 test probabilities (scores.npz from S2's final_model_report.py; rows are in
splits/isic_clinical_v2.csv test order, checked by reproducing the reported AUROC) with S1's binary tree
(trees/s1_binary_tree.py: TreeSHAP in log-odds of malignant, trained on ground-truth labels).

Strata as in eval/s1_error_lists.py: deferred = entropy > internal v2-val 75th percentile; decision threshold from
final_model_report.json. Then per stratum and per error type (TP / TN / FN / FP of the ResNet):
  - mean |SHAP| per concept group and per feature, and its share on C2-validated / partial / failed features
  - mean SHAP signed towards the ResNet's predicted class (+ = the tree's evidence agrees with the ResNet)
  - tree-ResNet agreement, and how well tree disagreement flags ResNet errors (AUROC) next to entropy

    python trees/s1_error_drivers.py --results ~/Downloads/s1_results_2026-10-23 \
        --tree trees/output/s1_binary_v0.2 --out trees/output/s1_binary_v0.2/error_drivers
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from feature_validity import best_status

ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {"A": "asymmetry", "B": "border", "C": "color", "D": "diameter", "T": "texture/surface",
               "S": "shape"}


def entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def resnet_frame(res):
    report = json.loads((res / "eval/output/stage_c/final/final_model_report.json").read_text())
    scores = np.load(res / "eval/output/stage_c/final/scores.npz")
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    test = v2[v2.split == "test"].reset_index(drop=True)
    p = scores["v2_test_prob"].astype(float)
    auc = roc_auc_score(test.label.astype(int), p)
    assert abs(auc - report["v2_test"]["auroc"]) < 1e-9, "scores.npz order does not match split v2 test"
    threshold = float(np.quantile(entropy(scores["v2_val_prob"]), 0.75))
    df = pd.DataFrame({"isic_id": test.isic_id, "label": test.label.astype(int), "source": test.source,
                       "diagnosis": test.diagnosis_3.fillna(test.diagnosis_2).fillna(test.diagnosis_1),
                       "p_resnet": p, "entropy": entropy(p)})
    df["pred_resnet"] = (df.p_resnet >= report["threshold_standard"]).astype(int)
    df["deferred"] = df.entropy > threshold
    df["wrong"] = df.pred_resnet != df.label
    df["outcome"] = np.select([(df.label == 1) & (df.pred_resnet == 1), (df.label == 0) & (df.pred_resnet == 0),
                               (df.label == 1) & (df.pred_resnet == 0)], ["TP", "TN", "FN"], "FP")
    df["stratum"] = np.where(df.deferred, "deferred", "confident") + "_" + np.where(df.wrong, "wrong", "right")
    return df, report, threshold


def main(opts):
    res = Path(opts.results).expanduser()
    tree = Path(opts.tree)
    df, report, threshold = resnet_frame(res)
    shap = pd.read_parquet(tree / "shap_values.parquet")
    feats = [c for c in shap.columns if c[:2] in {f"{g}_" for g in GROUP_NAMES}]
    tp = pd.read_csv(tree / "tree_predictions.csv")[["isic_id", "p_malignant", "pred"]]
    tp = tp.rename(columns={"p_malignant": "p_tree", "pred": "pred_tree"})
    d = df.merge(tp, on="isic_id").merge(shap[["isic_id"] + feats], on="isic_id")
    d["tree_agrees"] = d.pred_tree == d.pred_resnet

    sign = np.where(d.pred_resnet == 1, 1.0, -1.0)[:, None]
    absv = d[feats].abs()
    towards = d[feats].to_numpy() * sign
    groups = {g: [f for f in feats if f.startswith(f"{g}_")] for g in GROUP_NAMES}
    status = best_status(json.loads(Path(opts.validation).expanduser().read_text())).status
    vstat = pd.Series({f: status.get(f, "not tested") for f in feats})

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("stratum", "outcome"):
        for name, idx in d.groupby(key).groups.items():
            i = d.index.get_indexer(idx)
            r = {"by": key, "level": name, "n": len(i), "tree_agrees": d.tree_agrees.iloc[i].mean()}
            tot = absv.iloc[i].mean()
            for g, fs in groups.items():
                r[f"abs_{GROUP_NAMES[g]}"] = tot[fs].sum()
            for s in ("validated", "partial", "failed", "not tested"):
                r[f"share_{s}"] = tot[vstat[vstat == s].index].sum() / tot.sum()
            r["towards_pred_total"] = towards[i].sum(axis=1).mean()
            rows.append(r)
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(out / "strata_summary.csv", index=False)

    # Features whose signed contribution differs most between confident-wrong and confident-right
    cw, cr = (d.stratum == "confident_wrong").to_numpy(), (d.stratum == "confident_right").to_numpy()
    diff = pd.DataFrame({"feature": feats, "validity": vstat[feats].to_numpy(),
                         "towards_pred_confident_wrong": towards[cw].mean(axis=0),
                         "towards_pred_confident_right": towards[cr].mean(axis=0)})
    diff["difference"] = diff.towards_pred_confident_wrong - diff.towards_pred_confident_right
    diff = diff.reindex(diff.difference.abs().sort_values(ascending=False).index).round(3)
    diff.to_csv(out / "feature_difference_confident.csv", index=False)

    flag = {"entropy": roc_auc_score(d.wrong, d.entropy),
            "tree disagrees": roc_auc_score(d.wrong, (~d.tree_agrees).astype(int)),
            "|p_tree - p_resnet|": roc_auc_score(d.wrong, (d.p_tree - d.p_resnet).abs())}
    d.drop(columns=feats).to_csv(out / "test_rows.csv", index=False)

    lines = [f"# TreeSHAP on ResNet errors, internal test ({report['run']}; tree {tree.name})", "",
             f"n = {len(d)} test images with tree features (of {len(df)}); deferred = entropy > {threshold:.4f}; "
             f"ResNet threshold {report['threshold_standard']:.4f}.", "",
             "Mean |SHAP| by concept group and validity share; `towards_pred_total` = mean summed SHAP signed "
             "towards the ResNet's predicted class (negative = the tree's evidence points the other way).", "",
             summary.to_markdown(index=False), "",
             "AUROC for flagging ResNet errors:", "",
             pd.Series(flag).round(3).rename("auroc").to_frame().to_markdown(), "",
             "Features that differ most between confident-wrong and confident-right (signed towards prediction):",
             "", diff.head(opts.top).to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="unzipped s1_results folder")
    ap.add_argument("--tree", default="trees/output/s1_binary_v0.2")
    ap.add_argument("--validation", default=None, help="c2_validation.json (default: in --results)")
    ap.add_argument("--out", default="trees/output/s1_binary_v0.2/error_drivers")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    a.validation = a.validation or str(Path(a.results).expanduser() / "concepts/output/c2_validation.json")
    main(a)
