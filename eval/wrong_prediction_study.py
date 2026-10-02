"""
Explanations on wrong predictions (plan T5.4/T6.5; criterion C6) for the template baseline and Grad-CAM.

Strata on the test split: ResNet50 correct vs wrong x confident vs uncertain (uncertain = entropy above the
validation 75th percentile, i.e. the 25% deferral operating point from eval/baseline_report.py).

Questions:
  1. Do explanations lend the same support to wrong predictions as to correct ones?
     template: net support of the top-3 reasons, unanimous-support rate, confident-wording rate (no hedge)
     Grad-CAM: CAM mass in the lesion, pointing-game hits, deletion/insertion AUC
  2. Does any explanation signal carry information about errors? AUROC for "ResNet50 is wrong".
C6 asks that confident wording on incorrect predictions is no higher than on correct ones; reported with a
bootstrap 95% CI of the difference.

    python eval/wrong_prediction_study.py --inputs trees/output/explanation_inputs.jsonl \
        --gradcam eval/output/gradcam/gradcam_metrics.csv --predictions <run>/predictions.csv --out eval/output/c6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain"))
from template import render  # noqa: E402

STRENGTH = {"strong": 3, "moderate": 2, "weak": 1}


def bootstrap_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main(opts):
    preds = pd.read_csv(opts.predictions)
    threshold = float(preds[preds.split == "val"].entropy_of_expected.quantile(0.75))
    rows, texts = [], []
    with open(opts.inputs, encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item["split"] != "test":
                continue
            pl = item["llm_input"]
            exp = render(pl, threshold)
            signed = [STRENGTH[r["strength"]] * (1 if r["direction"] == "supports" else -1) for r in pl["top_concepts"]]
            pred = 1 if pl["prediction"] == "malignant" else 0
            rows.append({"isic_id": item["isic_id"], "source": item["source"], "mask_quality": item["mask_quality"],
                         "correct": pred == item["label"], "uncertain": pl["uncertainty"]["entropy_of_expected"] > threshold,
                         "entropy": pl["uncertainty"]["entropy_of_expected"], "surrogate_agrees": pl["surrogate_agrees"],
                         "net_support": float(np.sum(signed)), "n_supporting": int(np.sum(np.array(signed) > 0)),
                         "unanimous_support": bool(all(s > 0 for s in signed)),
                         "confident_wording": len(exp.hedges) == 0, "flagged_uncertain": "uncertain" in exp.hedges})
            texts.append({"isic_id": item["isic_id"], "label": item["label"], **pl, "explanation": exp.as_dict()})
    df = pd.DataFrame(rows)
    gc = pd.read_csv(opts.gradcam)
    df = df.merge(gc[["isic_id", "cam_in_lesion", "pointing_hit", "cam_entropy"] +
                     [c for c in ("deletion_auc", "insertion_auc", "random_deletion_auc") if c in gc]],
                  on="isic_id", how="left")
    df["stratum"] = np.where(df.correct, "correct", "wrong") + "/" + np.where(df.uncertain, "uncertain", "confident")

    cols = ["surrogate_agrees", "net_support", "unanimous_support", "confident_wording", "flagged_uncertain",
            "cam_in_lesion", "pointing_hit"] + [c for c in ("deletion_auc", "insertion_auc") if c in df]
    strata = df.groupby("stratum")[cols].mean().round(3)
    strata.insert(0, "n", df.groupby("stratum").size())

    wrong = (~df.correct).astype(int)
    detect = {"model entropy": df.entropy, "surrogate disagrees": (~df.surrogate_agrees).astype(int),
              "template: low net support": -df.net_support, "Grad-CAM: low CAM-in-lesion": -df.cam_in_lesion}
    if "deletion_auc" in df:
        detect["Grad-CAM: high deletion AUC"] = df.deletion_auc
    error_auroc = {k: float(roc_auc_score(wrong[v.notna()], v[v.notna()])) for k, v in detect.items()}

    c6 = {}
    for col in ("confident_wording", "unanimous_support", "net_support", "cam_in_lesion"):
        diff, ci = bootstrap_diff(df.loc[~df.correct, col].dropna(), df.loc[df.correct, col].dropna())
        c6[col] = {"wrong_minus_correct": diff, "ci95": ci}
    c5 = {"uncertain_cases": int(df.uncertain.sum()),
          "flagged_rate_among_uncertain": float(df.loc[df.uncertain, "flagged_uncertain"].mean())}

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "c6_per_image.csv", index=False)
    with open(out / "template_explanations_test.jsonl", "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps(t) + "\n")
    result = {"entropy_threshold_val_p75": threshold, "strata": strata.reset_index().to_dict(orient="records"),
              "error_detection_auroc": error_auroc, "c6_wrong_minus_correct": c6, "c5": c5,
              "by_source_wrong_confident_wording": df[~df.correct].groupby("source").confident_wording.mean().round(3).to_dict()}
    (out / "c6_results.json").write_text(json.dumps(result, indent=1))
    md = ["# Explanations on wrong predictions (C6), test split", "",
          f"Uncertain = entropy of expected > {threshold:.3f} (validation 75th percentile).", "",
          strata.to_markdown(), "", "## Does the explanation carry an error signal? (AUROC for 'ResNet50 wrong')", "",
          pd.Series(error_auroc).round(3).to_markdown(), "", "## C6: wrong minus correct (bootstrap 95% CI)", "",
          pd.DataFrame(c6).T.to_markdown(), "", f"C5: uncertain cases flagged in text: {c5['flagged_rate_among_uncertain']:.1%} "
          f"of {c5['uncertain_cases']}.", ""]
    (out / "c6_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--gradcam", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
