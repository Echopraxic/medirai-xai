"""
Darkness-contrast hypothesis (S1-9 finding 6, decision 2026-10-30): descriptive, nothing chosen from it.
Generalises Michael's eval/darkness_contrast_dsd.py (note 2026-10-30 section 2) to any model and to DDI.

Hypothesis: the shortcut "darker than the skin = benign, paler = malignant" (C_darkness_vs_skin) runs on contrast with
the surrounding skin, which shrinks on darker skin, and that would explain the low FST V-VI sensitivity.
  H1  C_darkness_vs_skin is lower on the darkest group than on the lightest, within each label.
  H2  per group, for malignant lesions, the feature's TreeSHAP (S1's v1.0 ground-truth tree, log-odds of malignant)
      differs between cancers the model misses and those it catches.
  H3  the model's own score tracks the feature within malignant lesions (Spearman), per group.
Intervals: 95% bootstrap over the feature table's `group` (lesion group; DDI has no patient id, so image), 2,000
resamples.

Sources:
  dsd  darker-skin dev set (selection data): groups FST IV vs V-VI. Missed = sigmoid(TTA margin) below the split-v2
       val threshold at 90% sensitivity (the sprint-2 operating point), from the run's tta_scores.npz.
  ddi  DDI (LOCKED, descriptive): groups I-II / III-IV / V-VI. Missed = `pred` 0 in the final model's DDI predictions
       from eval/final_locked_look.py (standard mode, threshold fit on split-v2 val). No model is scored here, but the
       read of DDI is still logged to looks.csv (--final required).

    python eval/darkness_contrast.py dsd --features features/output/external/dsd_features_v1.0.parquet \
        --tta uncertaintyNet-main/output/narval_stageb_kdv8f_mel_s3_20261030/run/tta_scores.npz \
        --out eval/output/darkness_contrast/dsd_kdv8f_mel_s3.json
    python eval/darkness_contrast.py ddi --features features/output/external/ddi_features_v1.0.parquet \
        --predictions eval/output/stage_c/locked_final/locked_final_kdv8f_mel_s3_ddi_predictions.csv \
        --final "darkness contrast on DDI, descriptive (note 2026-11-06 item 2)" \
        --out eval/output/darkness_contrast/ddi_kdv8f_mel_s3.json
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
FEATURE = "C_darkness_vs_skin"
N_BOOT, SEED = 2000, 0
GROUPS = {"dsd": ["IV", "V-VI"], "ddi": ["I-II", "III-IV", "V-VI"]}


def threshold_for_sensitivity(y, p, target=0.90):
    """Largest threshold whose sensitivity on (y, p) is >= target (as eval/selection_score.py, without torch)."""
    pos = np.sort(p[y == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))]) if len(pos) else 0.5


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


def load_dsd(d, opts):
    z = np.load(opts.tta, allow_pickle=True)
    p = pd.Series(1 / (1 + np.exp(-z["dsd_margin"])), index=z["dsd_ids"].astype(str))
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv").set_index("isic_id")
    pv = 1 / (1 + np.exp(-z["v2_val_margin"]))
    t = threshold_for_sensitivity(v2.loc[z["v2_val_ids"].astype(str), "label"].astype(int).values, pv)
    d["p"] = d.image_id.astype(str).map(p)
    d["caught"] = d.p >= t
    d["tone"] = np.where(d.dsd, "V-VI", "IV")
    return d, {"threshold_v2_val_sens90": t}


def load_ddi(d, opts):
    from external_test import log_look  # imports torch; only needed for the locked set
    if not opts.final:
        raise SystemExit("ddi is a locked external test set: give --final '<reason>' (logged to looks.csv)")
    print(f"locked set ddi: look #{log_look(Namespace(source='ddi', runs=[str(opts.predictions)], final=opts.final))}")
    pr = pd.read_csv(opts.predictions).set_index("image_id")
    d["p"] = d.image_id.map(pr.p_malignant)
    d["caught"] = d.image_id.map(pr.pred).eq(1)
    return d, {"predictions": str(opts.predictions), "missed_definition": "pred == 0 (standard mode)"}


def main(opts):
    d = pd.read_parquet(opts.features)
    d = d[d.extraction_error.fillna("") == ""].reset_index(drop=True)
    if opts.source == "ddi":
        import sys
        sys.path.insert(0, str(ROOT / "eval"))
    d, meta = (load_dsd if opts.source == "dsd" else load_ddi)(d, opts)
    unscored = int(d.p.isna().sum())
    d = d[d.p.notna()].reset_index(drop=True)  # an unscored image would otherwise count as missed

    model = xgb.XGBClassifier()
    model.load_model(opts.tree)
    feats = model.get_booster().feature_names
    sv = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(d[feats])
    d["shap"] = sv[:, feats.index(FEATURE)]

    groups = GROUPS[opts.source]
    lightest, darkest = groups[0], groups[-1]
    res = {"source": opts.source, **meta, "n_dropped_unscored": unscored,
           "n": {f"{g} label {l}": int(n) for (g, l), n in d.groupby(["tone", "label"]).size().items()},
           "H1_feature_by_tone": {}, "H2_shap_missed_vs_caught": {}, "H3_model_tracks_feature": {}}
    for lab in (0, 1):
        sub = d[(d.label == lab) & d.tone.isin([lightest, darkest])]
        stat = median_diff(lambda x: x.tone == darkest, lambda x: x.tone == lightest, FEATURE)
        res["H1_feature_by_tone"][f"label {lab}"] = {
            **{f"median_{g}": float(d.loc[(d.label == lab) & (d.tone == g), FEATURE].median()) for g in groups},
            f"diff_{darkest}_minus_{lightest}": stat(sub), "ci95": group_boot(sub, stat)}
    for g in groups:
        mal = d[(d.label == 1) & (d.tone == g)]
        stat = median_diff(lambda x: ~x.caught, lambda x: x.caught, "shap")
        missed = mal.loc[~mal.caught]
        res["H2_shap_missed_vs_caught"][g] = {
            "n_missed": len(missed), "n_caught": int(mal.caught.sum()), "sensitivity": float(mal.caught.mean()),
            "shap_median_missed": float(missed.shap.median()),
            "shap_median_caught": float(mal.loc[mal.caught, "shap"].median()),
            "feature_median_missed": float(missed[FEATURE].median()),
            "feature_median_caught": float(mal.loc[mal.caught, FEATURE].median()),
            "shap_diff_missed_minus_caught": stat(mal), "ci95": group_boot(mal, stat),
            "n_missed_pushed_benign": int((missed.shap < 0).sum()),
            "share_missed_pushed_benign": float((missed.shap < 0).mean())}
        rho = lambda x: float(spearmanr(x[FEATURE], x.p).statistic) if len(x) > 2 else float("nan")
        res["H3_model_tracks_feature"][g] = {"spearman_malignant": rho(mal), "ci95": group_boot(mal, rho)}
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(GROUPS))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--tree", type=Path, default=ROOT / "trees/output/s1_binary_v1.0/binary_gt.json")
    ap.add_argument("--tta", type=Path, help="dsd: the run's tta_scores.npz")
    ap.add_argument("--predictions", type=Path, help="ddi: final_locked_look.py DDI predictions csv")
    ap.add_argument("--final", default="", help="ddi: why this read is needed (logged)")
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args())
