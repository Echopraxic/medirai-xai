"""
Paired cluster-bootstrap comparison of two explained ResNet50 models on the same test images (split v2 test).

Images are resampled by lesion/patient group (`group_id` in the split file), the same resample is applied to both
models, and the difference (B - A) is reported with a 95% percentile interval. An interval that contains 0 means the
change is not distinguishable from resampling noise; it does not cover seed-to-seed training variance (one run each).

Inputs are `tree_predictions.csv` files from trees/train_trees.py (they carry the ResNet prediction, its entropy, the
surrogate prediction and the one-vs-all class), so surrogate agreement is measured against each model's own surrogate.

    python eval/paired_bootstrap.py --a trees/output_v1.0/tree_predictions.csv --b trees/output_v1.0_detv2/tree_predictions.csv \
        --out eval/output/paired_bootstrap_v1_vs_v2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 0


def metrics(d):
    wrong = d.resnet_pred != d.label
    agree = d.binary_surrogate_pred == d.resnet_pred
    mel = d[d.ova_class == "Melanoma"]
    out = {
        "accuracy": (~wrong).mean(),
        "AUROC": roc_auc_score(d.label, d.resnet_p_malignant),
        "malignant recall": ((d.resnet_pred == 1) & (d.label == 1)).sum() / max((d.label == 1).sum(), 1),
        "benign recall": ((d.resnet_pred == 0) & (d.label == 0)).sum() / max((d.label == 0).sum(), 1),
        "melanoma recall": (mel.resnet_pred == 1).mean() if len(mel) else np.nan,
        "surrogate agreement": agree.mean(),
        "surrogate agreement on ResNet-wrong": agree[wrong].mean() if wrong.any() else np.nan,
        "entropy AUROC for ResNet errors": roc_auc_score(wrong, d.entropy_of_expected) if 0 < wrong.sum() < len(d) else np.nan,
    }
    return out


def main(opts):
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v2.csv")[["isic_id", "group_id"]]
    cols = ["isic_id", "split", "label", "ova_class", "resnet_pred", "resnet_p_malignant", "entropy_of_expected",
            "binary_surrogate_pred"]
    a = pd.read_csv(opts.a)[cols]
    b = pd.read_csv(opts.b)[cols]
    a, b = (x[x.split == "test"].merge(split, on="isic_id") for x in (a, b))
    common = sorted(set(a.isic_id) & set(b.isic_id))
    assert len(common) == len(a) == len(b), "models must be scored on the same test images"
    a, b = (x.set_index("isic_id").loc[common].reset_index() for x in (a, b))
    assert (a.label.values == b.label.values).all()
    groups = a.group_id.values
    idx_by_group = pd.Series(np.arange(len(a))).groupby(groups).apply(np.array)
    keys = idx_by_group.index.to_numpy()
    rng = np.random.default_rng(SEED)
    base_a, base_b = metrics(a), metrics(b)
    diffs = {k: [] for k in base_a}
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rows = np.concatenate([idx_by_group.iloc[i] for i in pick])
        ma, mb = metrics(a.iloc[rows]), metrics(b.iloc[rows])
        for k in diffs:
            diffs[k].append(mb[k] - ma[k])
    report = {"n_test": len(a), "n_groups": len(keys), "n_boot": N_BOOT, "a": str(opts.a), "b": str(opts.b), "metrics": {}}
    lines = [f"# Paired bootstrap, B - A (n = {len(a)} test images, {len(keys)} groups, {N_BOOT} resamples)", "",
             f"A = `{opts.a}`; B = `{opts.b}`", "", "| metric | A | B | B - A | 95% CI | distinguishable from 0 |", "|---|---|---|---|---|---|"]
    for k in base_a:
        d = np.array(diffs[k])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        sig = "yes" if lo > 0 or hi < 0 else "no"
        report["metrics"][k] = {"A": base_a[k], "B": base_b[k], "diff": base_b[k] - base_a[k], "ci95": [lo, hi], "significant": sig}
        lines.append(f"| {k} | {base_a[k]:.3f} | {base_b[k]:.3f} | {base_b[k] - base_a[k]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "paired_bootstrap.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "paired_bootstrap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
