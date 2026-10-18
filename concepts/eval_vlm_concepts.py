"""
Score VLM concept answers (concepts/vlm_concepts.py) against Derm7pt expert labels, and cross-check the VLM's
A/B/C answers and the expert labels against our handcrafted features (concepts/derm7pt_features.py).

Agreement is Cohen's kappa (chance-corrected; quadratic-weighted for ordered answers) next to raw accuracy
and the majority-class accuracy, because models default to the majority answer.

    python concepts/eval_vlm_concepts.py concepts/output/vlm [--derm7pt data/derm7pt/release_v0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DERM7PT = ROOT / "data/derm7pt/release_v0"
FEATURES = ROOT / "concepts/output/derm7pt_clinic_features.parquet"
ORDERED = {"elevation": ["flat", "palpable", "nodular"],
           "asymmetry": ["symmetric", "asymmetric in one axis", "asymmetric in two axes"]}
COLLAPSE = {  # Derm7pt label -> the VLM's answer set
    "pigmentation": lambda v: "absent" if v == "absent" else ("irregular" if "irregular" in v else "regular"),
    "regression_structures": lambda v: "absent" if v == "absent" else "present",
    "vascular_structures": lambda v: "absent" if v == "absent" else "present",
}
FEATURE_CHECKS = {  # expert label (present/irregular = 1) -> our features expected to rise with it
    "blue_whitish_veil": ["C_frac_blue_gray", "C_deltaE_vs_skin"],
    "pigmentation": ["C_color_std", "C_color_entropy", "A_color_asym_deltaE"],
    "elevation": ["T_roughness", "T_glcm_contrast"],
}


def expert_labels(derm7pt=DERM7PT):
    m = pd.read_csv(Path(derm7pt) / "meta" / "meta.csv").set_index("case_num")
    for col, fn in COLLAPSE.items():
        m[col] = m[col].map(fn)
    return m


def score(y_true, y_pred, question):
    ok = y_pred.notna() & y_true.notna()
    t, p = y_true[ok], y_pred[ok]
    weights = "quadratic" if question in ORDERED else None
    if weights:
        order = {v: i for i, v in enumerate(ORDERED[question])}
        t, p = t.map(order), p.map(order)
    return {"n": int(ok.sum()), "accuracy": float((t == p).mean()),
            "majority_baseline": float(t.value_counts(normalize=True).iloc[0]),
            "kappa": float(cohen_kappa_score(t, p, weights=weights)) if t.nunique() > 1 else float("nan"),
            "vlm_answer_distribution": p.value_counts(normalize=True).round(3).to_dict()}


def main(folder, derm7pt=DERM7PT):
    folder = Path(folder)
    meta = expert_labels(derm7pt)
    feats = pd.read_parquet(FEATURES).set_index("case_num") if FEATURES.exists() else None
    report, rows = {}, []
    for run in sorted(p for p in folder.iterdir() if (p / "answers.jsonl").exists()):
        ans = pd.DataFrame([{"case_num": r["case_num"], **(r["answers"] or {})}
                            for r in map(json.loads, open(run / "answers.jsonl", encoding="utf-8"))]).set_index("case_num")
        res = {"parse_rate": float(ans.drop(columns=[], errors="ignore").notna().any(axis=1).mean())}
        for q in ans.columns:
            if q in meta.columns:
                res[q] = score(meta.loc[ans.index, q], ans[q], q)
                rows.append({"run": run.name, "question": q, **{k: v for k, v in res[q].items()
                                                                  if k != "vlm_answer_distribution"}})
        # A/B/C answers vs our features (no expert labels for these in Derm7pt)
        if feats is not None and run.name.endswith("__clinic"):
            j = ans.join(feats, how="inner")
            if len(j) > 30:
                res["abc_vs_features"] = {
                    "asymmetry~A_shape_asym_max": float(spearmanr(j.asymmetry.map({v: i for i, v in enumerate(ORDERED["asymmetry"])}),
                                                                  j.A_shape_asym_max, nan_policy="omit").statistic),
                    "border_irregular~B_radial_cv": float(roc_auc_score(j.border == "irregular", j.B_radial_cv))
                    if (j.border == "irregular").nunique() == 2 else None,
                    "n_colors~C_n_colors": float(spearmanr(j.colors.map(lambda c: len(set(c)) if isinstance(c, list) else np.nan),
                                                           j.C_n_colors, nan_policy="omit").statistic),
                }
        report[run.name] = res
    if feats is not None:  # our features vs expert labels on the clinical photos
        fx = {}
        j = meta.join(feats, how="inner")
        for label, cols in FEATURE_CHECKS.items():
            y = j[label].isin(["present", "irregular", "palpable", "nodular"]) if label != "elevation" else j[label] != "flat"
            fx[label] = {c: float(roc_auc_score(y, j[c])) for c in cols if j[c].notna().all() and y.nunique() == 2}
        report["our_features_vs_expert_auroc"] = fx
    (folder / "eval.json").write_text(json.dumps(report, indent=1))
    table = pd.DataFrame(rows).round(3)
    (folder / "eval.md").write_text("# VLM concept answers vs Derm7pt experts\n\n" + table.to_markdown(index=False)
                                    + "\n\nSee eval.json for answer distributions, A/B/C-vs-feature checks and our "
                                    "features vs expert labels.\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k == "our_features_vs_expert_auroc"
                      or isinstance(v, dict) and "abc_vs_features" in v}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(ROOT / "concepts/output/vlm"))
    ap.add_argument("--derm7pt", default=str(DERM7PT), help="Derm7pt release folder (contains meta/meta.csv)")
    opts = ap.parse_args()
    main(opts.folder, opts.derm7pt)
