"""
TreeSHAP on the binary surrogate -> concept-grouped top-3 reasons per prediction (plan T2.3, T2.4, T2.6).

- SHAP: shap.TreeExplainer, tree_path_dependent (exact for XGBoost), log-odds of "malignant".
- Direction is taken relative to ResNet50's predicted class: + supports the prediction, - argues against it.
- Concept grouping: SHAP summed within each group (A, B, C, D, T, S) before ranking, so three correlated
  color features cannot take all three slots. Top-3 groups by |grouped SHAP|.
- Strength buckets (strong/moderate/weak) are tertiles of |grouped SHAP| of top-3 entries on the
  validation split (fixed before looking at test).
- Each reason carries the single feature in the group that contributed most in the group's direction
  (feature_id), its value, and its percentile among benign training lesions (for phrasing).
- Stability: top-3 concept sets from bootstrap-refit surrogates vs the main model (mean Jaccard).

Writes explanation_inputs.jsonl (one structured input per val/test image; the LLM payload part has no
image, metadata or identifier) and shap_values.parquet.

    python trees/treeshap_top3.py --features features/output/feature_table_v0.1.parquet --trees trees/output
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

GROUP_NAMES = {"A": "asymmetry", "B": "border", "C": "color", "D": "diameter", "T": "texture/surface",
               "S": "shape"}


# Model features never cited as a reason's feature_id (no validated clinical wording yet); their SHAP still
# counts toward the group total.
NOT_PHRASED = {"C_n_colors"}


def representative(contrib: pd.Series) -> str:
    """Feature cited for a group: the largest contribution in the group's direction, skipping NOT_PHRASED."""
    phrased = contrib.drop(list(NOT_PHRASED), errors="ignore")
    return (phrased if len(phrased) else contrib).idxmax()


def grouped(shap_df, feats):
    groups = sorted({f[0] for f in feats})
    return pd.DataFrame({g: shap_df[[f for f in feats if f[0] == g]].sum(1) for g in groups})


def strength_bucket(v, thresholds):
    """|grouped SHAP| -> strong/moderate/weak, cut at the validation tertiles (t1, t2)."""
    v, (t1, t2) = abs(v), thresholds
    return "strong" if v >= t2 else ("moderate" if v >= t1 else "weak")


def benign_percentile(ref_sorted, value):
    return float(np.searchsorted(ref_sorted, value) / max(len(ref_sorted), 1) * 100)


def case_reasons(group_row, signed_row, values, feats, ref, thresholds, not_cited=frozenset(), not_encoded=None):
    """Top-3 concept groups of one case -> reason dicts. group_row: grouped signed SHAP (index = group letters);
    signed_row: per-feature SHAP toward the predicted class; values: feature values; ref: sorted benign train values.
    not_cited (faithfulness filter): features never cited as a reason; a group with no citable member is skipped and
    the next group takes its slot (its SHAP still counts in every group total). not_encoded (labelling): when given, each
    reason gets model_encoded = False if the explained ResNet does not represent the cited feature."""
    reasons = []
    for grp in group_row.abs().sort_values(ascending=False, kind="stable").index:
        if len(reasons) == 3:
            break
        gval = group_row[grp]
        members = [f for f in feats if f[0] == grp and f not in not_cited]
        if not members:
            continue
        fid = representative(signed_row[members] * np.sign(gval))
        value = float(values[fid])
        reasons.append({"concept": GROUP_NAMES[grp], "group": grp, "feature_id": fid,
                        "direction": "supports" if gval > 0 else "argues_against",
                        "strength": strength_bucket(gval, thresholds), "grouped_shap": round(float(gval), 4),
                        "feature_value": round(value, 4), "benign_percentile": round(benign_percentile(ref[fid], value), 1)})
        if not_encoded is not None:
            reasons[-1]["model_encoded"] = fid not in not_encoded
    return reasons


def top3_sets(signed_groups):
    return [frozenset(r.abs().nlargest(3).index) for _, r in signed_groups.iterrows()]


def main(opts):
    trees = Path(opts.trees)
    feats = json.loads((trees / "features_used.json").read_text())
    not_cited = frozenset(json.loads(Path(opts.not_cited).read_text())["features"]) if opts.not_cited else frozenset()
    not_encoded = (frozenset(json.loads(Path(opts.label_not_encoded).read_text())["features"])
                   if opts.label_not_encoded else None)
    table = pd.read_parquet(opts.features)
    tp = pd.read_csv(trees / "tree_predictions.csv")
    df = tp.merge(table[["isic_id"] + feats], on="isic_id")
    df = df[(df.mask_quality.notna())].reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(trees / "binary_surrogate.json")
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = pd.DataFrame(explainer.shap_values(df[feats]), columns=feats)
    sv.insert(0, "isic_id", df.isic_id)
    sv.to_parquet(trees / "shap_values.parquet", index=False)

    sign = np.where(df.resnet_pred == 1, 1.0, -1.0)            # toward ResNet50's predicted class
    signed = sv[feats].mul(sign, axis=0)
    g = grouped(signed, feats)

    # strength tertiles from validation top-3 magnitudes
    val_mag = np.concatenate([r.abs().nlargest(3).values for _, r in g[df.split == "val"].iterrows()])
    t1, t2 = np.quantile(val_mag, [1 / 3, 2 / 3])

    # benign training reference for percentiles
    train_tab = table[(table.split == "train") & (table.label == 0)]
    ref = {f: np.sort(train_tab[f].dropna().values) for f in feats}

    # stability: bootstrap-refit surrogates with the same hyper-parameters
    report = json.loads((trees / "tree_report.json").read_text())
    params = {k: v for k, v in report["binary_surrogate"]["params"].items() if k != "best_iteration"}
    n_est = report["binary_surrogate"]["params"]["best_iteration"] + 1
    full = tp.merge(table[["isic_id", "extraction_error"] + feats], on="isic_id")
    train = full[(full.split == "train") & (full.mask_quality != "fail") & (full.extraction_error.fillna("") == "")]
    test_idx = np.where(df.split == "test")[0]
    main_sets = top3_sets(g.iloc[test_idx])
    jaccards = []
    rng = np.random.default_rng(0)
    for b in range(opts.bootstraps):
        boot = train.iloc[rng.integers(0, len(train), len(train))]
        pos = boot.resnet_pred.mean()
        mb = xgb.XGBClassifier(n_estimators=n_est, learning_rate=0.03, scale_pos_weight=(1 - pos) / pos,
                               tree_method="hist", random_state=b, n_jobs=6, **params)
        mb.fit(boot[feats], boot.resnet_pred)
        svb = pd.DataFrame(shap.TreeExplainer(mb).shap_values(df.iloc[test_idx][feats]), columns=feats)
        sets_b = top3_sets(grouped(svb.mul(sign[test_idx], axis=0), feats))
        jaccards.append(np.mean([len(a & c) / len(a | c) for a, c in zip(main_sets, sets_b)]))

    lines = []
    for i, r in df[df.split.isin(["val", "test"])].iterrows():
        reasons = case_reasons(g.loc[i], signed.loc[i], df.loc[i], feats, ref, (t1, t2), not_cited, not_encoded)
        # ungrouped top-3 features (raw SHAP toward the prediction): input for the F-raw encoding (explain/encodings.py)
        top_feats = signed.loc[i, feats].abs().nlargest(3).index
        raw_top = [{"feature_id": f, "group": f[0], "shap": round(float(signed.loc[i, f]), 4),
                    "feature_value": round(float(df.loc[i, f]), 4),
                    "benign_percentile": round(benign_percentile(ref[f], df.loc[i, f]), 1)}
                   for f in top_feats]
        payload = {
            "prediction": "malignant" if r.resnet_pred == 1 else "benign",
            "probability_malignant": round(float(r.resnet_p_malignant), 3),
            "uncertainty": {"entropy_of_expected": round(float(r.entropy_of_expected), 4)},
            "surrogate_agrees": bool(r.binary_surrogate_pred == r.resnet_pred),
            # ground-truth feature tree's probability of the ResNet50's predicted class: an independent second
            # opinion that flags confident errors better than entropy (eval/error_signals.py)
            "feature_model_p_predicted": round(float(r.binary_gt_p if r.resnet_pred == 1 else 1 - r.binary_gt_p), 3),
            "top_concepts": reasons,
            "top_features": raw_top,
            "audience": "dermatologist",
        }
        lines.append({"isic_id": r.isic_id, "split": r.split, "label": int(r.label), "mask_quality": r.mask_quality,
                      "source": r.source, "llm_input": payload})
    with open(trees / f"explanation_inputs{opts.suffix}.jsonl", "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")

    summary = {"strength_thresholds_val": [float(t1), float(t2)],
               "top3_bootstrap_jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
               "top3_bootstrap_jaccard_per_refit": [float(j) for j in jaccards],
               "group_frequency_in_top3_test": pd.Series([x for s in main_sets for x in s]).map(GROUP_NAMES)
               .value_counts(normalize=True).round(3).to_dict(),
               "n_inputs": len(lines)}
    (trees / f"treeshap_summary{opts.suffix}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--trees", required=True)
    ap.add_argument("--bootstraps", type=int, default=5)
    ap.add_argument("--not_cited", default="", help="faithfulness filter: not_cited.json from interpret/explanation_features.py")
    ap.add_argument("--label_not_encoded", default="",
                    help="labelling (pipeline default since 2026-10-21): not_cited.json; reasons get model_encoded")
    ap.add_argument("--suffix", default="", help="output suffix, e.g. _filtered (explanation_inputs<suffix>.jsonl)")
    main(ap.parse_args())
