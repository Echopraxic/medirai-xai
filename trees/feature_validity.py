"""
Attach C2 concept-validity status to a tree's TreeSHAP importance table (S1-5 / S1-9 wording).

For each handcrafted feature, takes its best confirmatory result in concepts/output/c2_validation.json
(validated > partial > failed; the highest-AUROC row within that status) and suggests how to name the
feature in the write-up:
  validated   may be named after the clinical concept it validated against
  partial     name as the image property, say "partly tracks <concept>"
  failed      name as the image property only (e.g. "image colour entropy"), never as a clinical finding
  not tested  no expert label to test against (shape, border, diameter); name as the image property

    python trees/feature_validity.py --importance trees/output/s1_binary_v0.2/importance.csv \
        --validation ~/Downloads/s1_results_2026-10-23/concepts/output/c2_validation.json \
        --out trees/output/s1_binary_v0.2/importance_validity.md
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RANK = {"validated": 0, "partial": 1, "failed": 2}


def best_status(validation):
    rows = pd.DataFrame(validation["confirmatory"])
    rows = rows[rows.kind == "handcrafted"].assign(rank=lambda d: d.status.map(RANK))
    best = rows.sort_values(["rank", "auroc"], ascending=[True, False]).groupby("candidate").head(1)
    return best.set_index("candidate")[["status", "concept", "dataset", "auroc", "diagnosis_confounded"]]


def wording(row):
    if row.status == "validated":
        return f"{row.concept} (validated, AUROC {row.auroc:.2f})"
    if row.status == "partial":
        return f"image property; partly tracks {row.concept} (AUROC {row.auroc:.2f})"
    if row.status == "failed":
        return f"image property only; failed as {row.concept} (AUROC {row.auroc:.2f})"
    return "image property (no expert label to test)"


def main(opts):
    imp = pd.read_csv(opts.importance)
    imp = imp[imp.level == "feature"].copy()
    status = best_status(json.loads(Path(opts.validation).expanduser().read_text()))
    imp = imp.join(status, on="name")
    imp["status"] = imp.status.fillna("not tested")
    imp["suggested_wording"] = imp.apply(wording, axis=1)
    imp["confounded"] = imp.diagnosis_confounded.map({True: "yes", False: ""}).fillna("")
    top = imp.sort_values("mean_abs_shap", ascending=False).head(opts.top)

    share = imp.groupby("status").mean_abs_shap.sum() / imp.mean_abs_shap.sum()
    lines = [f"# Feature validity for {opts.importance}", "",
             "Share of total mean |SHAP| by validity status:", "",
             share.round(3).rename("share").to_frame().to_markdown(), "",
             f"Top {opts.top} drivers:", "",
             top[["name", "group", "mean_abs_shap", "direction", "status", "confounded", "suggested_wording"]]
             .round(3).to_markdown(index=False), ""]
    text = "\n".join(lines)
    if opts.out:
        Path(opts.out).write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--importance", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out")
    ap.add_argument("--top", type=int, default=15)
    main(ap.parse_args())
