"""
Darkness-contrast hypothesis (S1-9 finding 6, decision 2026-10-30) on the darker-skin dev set: descriptive, selection
data only, nothing chosen from it.

Shatha's hypothesis: the shortcut "darker than the skin = benign, paler = malignant" (C_darkness_vs_skin) runs on
contrast with the surrounding skin, which shrinks on darker skin, and that would explain the low FST V-VI sensitivity.
  H1  C_darkness_vs_skin is lower on FST V-VI than on FST IV, within each label.
  H2  for malignant FST V-VI lesions, the feature's TreeSHAP (S1's v1.0 ground-truth tree, log-odds of malignant)
      differs between cancers the final model misses and those it catches.
  H3  the final ResNet's own score tracks the feature within malignant lesions (Spearman), per FST group.
Missed = sigmoid(TTA margin) below the v2-val threshold at 90% sensitivity (the sprint-2 operating point,
eval/sprint2_analysis.py). Intervals: 95% bootstrap over lesion groups (group_id), 2,000 resamples.

Inputs: features/output/external/dsd_features_v1.0.parquet, trees/output/s1_binary_v1.0/binary_gt.json,
<final run>/tta_scores.npz, splits/isic_clinical_v2.csv.

    python eval/darkness_contrast_dsd.py   ->  eval/output/darkness_contrast/dsd.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "trees")]
from selection_score import threshold_for_sensitivity  # noqa: E402
from train_trees import feature_columns  # noqa: E402

RUN = ROOT / "uncertaintyNet-main/output/narval_stageb_kdv6_mel_s1_20261022/run"
TREE = ROOT / "trees/output/s1_binary_v1.0"
FEATURE = "C_darkness_vs_skin"
N_BOOT, SEED = 2000, 0


def group_boot(df, stat, seed=SEED):
    """95% interval of stat(df) over resampled lesion groups."""
    rng = np.random.default_rng(seed)
    groups = df.group.unique()
    by = {g: idx for g, idx in df.groupby("group").indices.items()}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([by[g] for g in rng.choice(groups, len(groups))])
        v = stat(df.iloc[pick])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [None, None]


def median_diff(a_mask, b_mask, col):
    return lambda d: float(d.loc[a_mask(d), col].median() - d.loc[b_mask(d), col].median())


def main():
    d = pd.read_parquet(ROOT / "features/output/external/dsd_features_v1.0.parquet")
    d = d[d.extraction_error.fillna("") == ""].reset_index(drop=True)
    z = np.load(RUN / "tta_scores.npz", allow_pickle=True)
    p = pd.Series(1 / (1 + np.exp(-z["dsd_margin"])), index=z["dsd_ids"].astype(str))
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv").set_index("isic_id")
    pv = 1 / (1 + np.exp(-z["v2_val_margin"]))
    t = threshold_for_sensitivity(v2.loc[z["v2_val_ids"].astype(str), "label"].astype(int).values, pv)
    d["p"] = d.image_id.astype(str).map(p)
    d["caught"] = d.p >= t
    d["fst"] = np.where(d.dsd, "V-VI", "IV")

    model = xgb.XGBClassifier()
    model.load_model(TREE / "binary_gt.json")
    feats = model.get_booster().feature_names or feature_columns(d)
    sv = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(d[feats])
    d["shap"] = sv[:, feats.index(FEATURE)]

    res = {"threshold_v2_val_sens90": t, "n": {f"{f} label {l}": int(n) for (f, l), n in d.groupby(["fst", "label"]).size().items()},
           "H1_feature_by_fst": {}, "H2_shap_missed_vs_caught": {}, "H3_resnet_tracks_feature": {}}
    for lab in (0, 1):
        sub = d[d.label == lab]
        dark, light = (lambda x: x.dsd), (lambda x: ~x.dsd)
        stat = median_diff(dark, light, FEATURE)
        res["H1_feature_by_fst"][f"label {lab}"] = {
            "median_IV": float(sub.loc[~sub.dsd, FEATURE].median()), "median_V-VI": float(sub.loc[sub.dsd, FEATURE].median()),
            "diff_V-VI_minus_IV": stat(sub), "ci95": group_boot(sub, stat)}
    for fst in ("IV", "V-VI"):
        mal = d[(d.label == 1) & (d.fst == fst)]
        stat = median_diff(lambda x: ~x.caught, lambda x: x.caught, "shap")
        res["H2_shap_missed_vs_caught"][fst] = {
            "n_missed": int((~mal.caught).sum()), "n_caught": int(mal.caught.sum()),
            "sensitivity": float(mal.caught.mean()),
            "shap_median_missed": float(mal.loc[~mal.caught, "shap"].median()),
            "shap_median_caught": float(mal.loc[mal.caught, "shap"].median()),
            "feature_median_missed": float(mal.loc[~mal.caught, FEATURE].median()),
            "feature_median_caught": float(mal.loc[mal.caught, FEATURE].median()),
            "shap_diff_missed_minus_caught": stat(mal), "ci95": group_boot(mal, stat),
            "share_missed_pushed_benign": float((mal.loc[~mal.caught, "shap"] < 0).mean())}
        rho = lambda x: float(spearmanr(x[FEATURE], x.p).statistic) if len(x) > 2 else float("nan")
        res["H3_resnet_tracks_feature"][fst] = {"spearman_malignant": rho(mal), "ci95": group_boot(mal, rho)}
    out = ROOT / "eval/output/darkness_contrast"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dsd.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
