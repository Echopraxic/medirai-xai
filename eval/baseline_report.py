"""
Baseline report for one or more exported runs (eval/export_predictions.py) on the grouped split.

Per run: test accuracy, balanced accuracy, AUROC, ECE (positive class and top-label), Brier, per-source and
per-diagnosis accuracy, and an entropy-of-expected deferral curve. The "uncertain" operating point used
downstream (C6 strata) defers the 25% most uncertain cases, with the threshold fixed on the validation set.

    python eval/baseline_report.py --runs det=s2_local/resnet50_det/run_X vll_std=... --out eval/output/baseline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uncertaintyNet-main"))
import utils  # noqa: E402

DEFER_FRACTION = 0.25   # top-25% entropy on validation defines the "uncertain/deferred" threshold
SOURCE_SHORT = {"MILK study team": "MILK", "Memorial Sloan Kettering Cancer Center": "MSKCC",
                "Federal University of Espírito Santo (UFES)": "UFES"}


def short_source(s):
    for k, v in SOURCE_SHORT.items():
        if isinstance(s, str) and s.startswith(k[:20]):
            return v
    return "other"


def metrics(df):
    y, p, pred = df.label.values, df.p_malignant.values, df.pred.values
    probs = np.stack([1 - p, p], 1)
    out = {"n": len(df), "accuracy": float((pred == y).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "ece_pos": float(utils.expected_calibration_error(y, p)),
           "ece_top_label": float(utils.top_label_calibration_error(probs, y)),
           "brier": float(np.mean((p - y) ** 2))}
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return out


def deferral_curve(test, unc="entropy_of_expected"):
    order = test.sort_values(unc)
    correct = (order.pred == order.label).values
    rows = []
    for frac in (0.0, 0.1, 0.2, 0.25, 0.3, 0.374, 0.4, 0.5):
        keep = int(round(len(order) * (1 - frac)))
        rows.append({"deferred": frac, "retained_accuracy": float(correct[:keep].mean())})
    return rows


def report_run(name, run_dir):
    pred = pd.read_csv(run_dir / "predictions.csv")
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v1.csv")[["isic_id", "attribution"]]
    pred = pred.merge(split, on="isic_id", how="left")
    pred["source"] = pred.attribution.map(short_source)
    val, test = pred[pred.split == "val"], pred[pred.split == "test"]
    threshold = float(val.entropy_of_expected.quantile(1 - DEFER_FRACTION))
    test = test.assign(deferred=test.entropy_of_expected > threshold)
    res = {"run": name, "dir": str(run_dir), "meta": json.loads((run_dir / "predictions_meta.json").read_text()),
           "test": metrics(test), "val": metrics(val),
           "test_by_source": {s: metrics(g) for s, g in test.groupby("source") if len(g) >= 20},
           "test_without_mskcc": metrics(test[test.source != "MSKCC"]),
           "test_by_diagnosis": test.groupby("diagnosis_3").apply(
               lambda g: pd.Series({"n": len(g), "accuracy": (g.pred == g.label).mean()}), include_groups=False)
               .query("n >= 10").round(3).to_dict(orient="index"),
           "deferral_curve": deferral_curve(test),
           "val_entropy_threshold": threshold,
           "test_deferred_fraction": float(test.deferred.mean()),
           "test_retained_accuracy": float((test[~test.deferred].pred == test[~test.deferred].label).mean()),
           "test_deferred_accuracy": float((test[test.deferred].pred == test[test.deferred].label).mean())}
    wrong = test[test.pred != test.label]
    res["overconfidence"] = {"mean_max_softmax_correct": float(test[test.pred == test.label].max_softmax.mean()),
                             "mean_max_softmax_wrong": float(wrong.max_softmax.mean()),
                             "wrong_with_conf_ge_0.9": int((wrong.max_softmax >= 0.9).sum()), "n_wrong": len(wrong)}
    log = run_dir / "log.csv"
    if log.exists():
        res["training_log"] = pd.read_csv(log).to_dict(orient="list")
    return res


def to_markdown(results):
    lines = ["# Provisional baseline on the grouped split (isic_clinical_v1)", "",
             "Local RTX 3050 runs (batch 32 x 4 accumulation, fp16 autocast, 768-px image cache). "
             "Provisional: the official numbers are S1's re-run of the same scripts on the VM/cluster.", ""]
    cols = ["accuracy", "balanced_accuracy", "auroc", "ece_pos", "ece_top_label", "brier"]
    lines += ["## Test set", "", "| run | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in results:
        t = r["test"]
        lines.append(f"| {r['run']} | {t['n']} | " + " | ".join(f"{t[c]:.3f}" for c in cols) + " |")
    lines += ["", "## Test by source (label/source confound: MSKCC is 99.5% benign)", "",
              "| run | source | n | accuracy | balanced acc | AUROC |", "|---|---|---|---|---|---|"]
    for r in results:
        for s, t in {**r["test_by_source"], "all but MSKCC": r["test_without_mskcc"]}.items():
            lines.append(f"| {r['run']} | {s} | {t['n']} | {t['accuracy']:.3f} | {t['balanced_accuracy']:.3f} | {t['auroc']:.3f} |")
    lines += ["", "## Deferral by entropy of expected (test)", "",
              "| run | " + " | ".join(f"defer {d['deferred']:.0%}" for d in results[0]["deferral_curve"]) + " |",
              "|---|" + "---|" * len(results[0]["deferral_curve"])]
    for r in results:
        lines.append(f"| {r['run']} | " + " | ".join(f"{d['retained_accuracy']:.3f}" for d in r["deferral_curve"]) + " |")
    lines += ["", "## Overconfidence", "", "| run | max-softmax correct | max-softmax wrong | wrong with conf >= 0.9 |",
              "|---|---|---|---|"]
    for r in results:
        o = r["overconfidence"]
        lines.append(f"| {r['run']} | {o['mean_max_softmax_correct']:.3f} | {o['mean_max_softmax_wrong']:.3f} | "
                     f"{o['wrong_with_conf_ge_0.9']}/{o['n_wrong']} |")
    lines += ["", "## Test accuracy by diagnosis (n >= 10)", ""]
    diags = sorted(results[0]["test_by_diagnosis"])
    lines += ["| diagnosis | n | " + " | ".join(r["run"] for r in results) + " |", "|---|---|" + "---|" * len(results)]
    for d in diags:
        n = int(results[0]["test_by_diagnosis"][d]["n"])
        lines.append(f"| {d} | {n} | " + " | ".join(f"{r['test_by_diagnosis'].get(d, {}).get('accuracy', float('nan')):.3f}"
                                                      for r in results) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_folder (relative to uncertaintyNet-main/output)")
    ap.add_argument("--out", required=True)
    opts = ap.parse_args()
    results = []
    for item in opts.runs:
        name, rel = item.split("=", 1)
        results.append(report_run(name, ROOT / "uncertaintyNet-main" / "output" / rel))
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_results.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "baseline_report.md").write_text(to_markdown(results), encoding="utf-8")
    print(to_markdown(results))
