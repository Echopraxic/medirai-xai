"""
Per-image uncertainty + multi-model agreement/deferral table, keyed by isic_id (plan S1-4 -> T4.2, T6.5).

Inputs are per-image prediction CSVs with at least isic_id, split, label, pred, p_malignant: the
predictions.csv of each uncertaintyNet run (eval/export_predictions.py) and, optionally, S1's binary
tree (trees/s1_binary_tree.py -> tree_predictions.csv). Only images present in every input are kept.

Per model <m>:   <m>_pred, <m>_p_malignant, and where the input has them <m>_entropy_of_expected,
                 <m>_expected_entropy, <m>_mutual_information, <m>_variational_variance (= MC variance
                 of p(malignant); variational runs only)
Combined:        models_agree         every model predicts the same class
                 disagreeing_models   models whose prediction differs from the primary model's ("" if none)
                 high_entropy         primary entropy-of-expected above the validation threshold
                                      (top --defer-fraction of validation, as in eval/baseline_report.py)
                 deferred             high_entropy or not models_agree
                 defer_reason         "high entropy", "models disagreed (<models>)", or both

    python eval/uncertainty_table.py --primary det \
        --runs det=uncertaintyNet-main/output/<TAG>/resnet50_det/run/predictions.csv \
               vll_std=.../resnet50_vll_std/run/predictions.csv vll_elbo=.../resnet50_vll_elbo/run/predictions.csv \
               tree=trees/output/s1_binary_v0.2/tree_predictions.csv \
        --out eval/output/<TAG>/uncertainty
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEFER_FRACTION = 0.25
UNC_COLS = ("entropy_of_expected", "expected_entropy", "mutual_information")


def load_model(name, path):
    p = pd.read_csv(path)
    out = p[["isic_id", "split", "label", "pred", "p_malignant"]].copy()
    for c in UNC_COLS:
        if c in p:
            out[c] = p[c]
    if "mc_std_p_malignant" in p and (p.mc_std_p_malignant > 0).any():
        out["variational_variance"] = p.mc_std_p_malignant ** 2
    keep = ["isic_id", "split", "label"]
    return out.rename(columns={c: f"{name}_{c}" for c in out.columns if c not in keep})


def build(runs, primary, defer_fraction=DEFER_FRACTION):
    """runs: {name: csv path}, in column order. Returns (table, summary)."""
    names = list(runs)
    if primary not in runs:
        raise ValueError(f"--primary {primary!r} is not one of the runs {names}")
    df = None
    for name, path in runs.items():
        m = load_model(name, path)
        df = m if df is None else df.merge(m, on=["isic_id", "split", "label"], how="inner")
    if f"{primary}_entropy_of_expected" not in df:
        raise ValueError(f"primary model {primary!r} has no entropy_of_expected column")

    preds = df[[f"{n}_pred" for n in names]].to_numpy()
    df["models_agree"] = (preds == preds[:, :1]).all(1)
    ref = df[f"{primary}_pred"].to_numpy()
    df["disagreeing_models"] = [",".join(n for n, p in zip(names, row) if p != r) for row, r in zip(preds, ref)]

    ent = df[f"{primary}_entropy_of_expected"]
    threshold = float(ent[df.split == "val"].quantile(1 - defer_fraction))
    df["high_entropy"] = ent > threshold
    df["deferred"] = df.high_entropy | ~df.models_agree
    df["defer_reason"] = [
        "; ".join(r for r in (("high entropy" if h else ""), (f"models disagreed ({d})" if d else "")) if r)
        for h, d in zip(df.high_entropy, df.disagreeing_models)]

    test = df[df.split == "test"]
    correct = test[f"{primary}_pred"] == test.label
    summary = {
        "models": names, "primary": primary, "n": len(df), "n_test": len(test),
        "defer_fraction_on_val": defer_fraction, "entropy_threshold": threshold,
        "test_deferred_fraction": float(test.deferred.mean()),
        "test_deferred_by": {"high_entropy_only": float((test.high_entropy & test.models_agree).mean()),
                             "disagreement_only": float((~test.high_entropy & ~test.models_agree).mean()),
                             "both": float((test.high_entropy & ~test.models_agree).mean())},
        "test_primary_accuracy": {"all": float(correct.mean()),
                                  "retained": float(correct[~test.deferred].mean()),
                                  "deferred": float(correct[test.deferred].mean())},
        "test_disagreement_counts": test.disagreeing_models[test.disagreeing_models != ""].value_counts().to_dict(),
    }
    return df, summary


def main(opts):
    runs = dict(r.split("=", 1) for r in opts.runs)
    df, summary = build(runs, opts.primary or next(iter(runs)), opts.defer_fraction)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "uncertainty_table.csv", index=False)
    (out / "uncertainty_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path/to/predictions.csv, one per model")
    ap.add_argument("--primary", default=None, help="model whose entropy and prediction are the reference "
                                                    "(default: the first run)")
    ap.add_argument("--defer-fraction", type=float, default=DEFER_FRACTION)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
