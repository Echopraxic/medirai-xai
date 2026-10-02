"""
Learning curve for the ResNet50 baseline: test metrics vs share of training lesion groups.

Runs are uncertaintyNet folders with exported predictions (eval/export_predictions.py), named
<label>=<run folder> where label is the training fraction (e.g. 0.25). Several runs per fraction are
averaged and their spread reported. Per-class accuracy shows which classes are still data-limited.

    python eval/learning_curve.py --out eval/output/learning_curve \
        --runs 0.25=s2_local/lc/f025_s0/run 0.25=s2_local/lc/f025_s1/run 1.0=s2_local/resnet50_det/run_X ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from build_table import OVA_MAP  # noqa: E402


def run_metrics(run_dir):
    p = pd.read_csv(run_dir / "predictions.csv")
    t = p[p.split == "test"].copy()
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v1.csv")[["isic_id", "attribution"]]
    t = t.merge(split, on="isic_id")
    t["cls"] = t.diagnosis_3.map(OVA_MAP).fillna("Other")
    no_mskcc = t[~t.attribution.str.startswith("Memorial")]
    out = {"accuracy": (t.pred == t.label).mean(), "balanced_accuracy": balanced_accuracy_score(t.label, t.pred),
           "auroc": roc_auc_score(t.label, t.p_malignant),
           "auroc_without_mskcc": roc_auc_score(no_mskcc.label, no_mskcc.p_malignant)}
    for c, g in t.groupby("cls"):
        out[f"acc_{c}"] = (g.pred == g.label).mean()
    log = pd.read_csv(run_dir / "log.csv")
    out["train_acc_last"] = log["Train Accuracy"].iloc[-1]
    out["val_acc_best"] = log["Val Accuracy"].max()
    return out


def main(opts):
    rows = []
    for item in opts.runs:
        frac, rel = item.split("=", 1)
        rows.append({"fraction": float(frac), "run": rel, **run_metrics(ROOT / "uncertaintyNet-main" / "output" / rel)})
    df = pd.DataFrame(rows)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "learning_curve_runs.csv", index=False)
    metrics = [c for c in df.columns if c not in ("fraction", "run")]
    mean = df.groupby("fraction")[metrics].mean()
    spread = df.groupby("fraction")[metrics].agg(lambda x: (x.max() - x.min()) / 2 if len(x) > 1 else np.nan)
    n_train = {f: int(round(5551 * f)) for f in mean.index}
    # log-linear slope per doubling of the training data (from the means)
    x = np.log2(mean.index.values)
    slope = {m: float(np.polyfit(x, mean[m].values, 1)[0]) for m in metrics} if len(mean) >= 2 else {}
    md = ["# Learning curve (ResNet50, deterministic, grouped split)", "",
          "Mean over runs per fraction; ± = half the range across runs (subsample seeds, or training seeds at 100%).", ""]
    show = ["accuracy", "balanced_accuracy", "auroc", "auroc_without_mskcc", "acc_Melanoma", "acc_SK", "acc_Nevus",
            "acc_BCC", "acc_SCC", "acc_Other", "train_acc_last", "val_acc_best"]
    head = "| metric | " + " | ".join(f"{f:.0%} (~{n_train[f]} imgs)" for f in mean.index) + " | change per doubling |"
    md += [head, "|---|" + "---|" * (len(mean.index) + 1)]
    for m in show:
        cells = [f"{mean.loc[f, m]:.3f}" + (f" ± {spread.loc[f, m]:.3f}" if not np.isnan(spread.loc[f, m]) else "")
                 for f in mean.index]
        md.append(f"| {m} | " + " | ".join(cells) + f" | {slope.get(m, float('nan')):+.3f} |")
    (out / "learning_curve.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "learning_curve.json").write_text(json.dumps({"slope_per_doubling": slope, "runs": rows}, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
