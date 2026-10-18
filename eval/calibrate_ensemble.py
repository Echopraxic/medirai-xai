"""
Post-hoc classifier improvements that need no retraining, all fit on the validation split and reported on test:

  1. operating point   threshold on p(malignant) chosen on val for a target malignant sensitivity (default 0.90)
  2. ensemble          mean of the (calibrated) p(malignant) of several exported runs
  3. temperature       one temperature per run, fit by NLL on val, applied to the run's log-odds

Every variant is compared with the reference (first run, uncalibrated, threshold 0.5) by a paired cluster bootstrap
over lesion groups (`group_id` in split v2), the same resampling as eval/paired_bootstrap.py.

    python eval/calibrate_ensemble.py --runs det=rorqual_v2_lr3e-4_20261014/resnet50_det/run ... \
        --out eval/output/calibrate_ensemble
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 0
EPS = 1e-6
N_BINS = 15


def log_odds(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_temperature(p_val, y_val):
    z = log_odds(p_val)

    def nll(t):
        q = np.clip(1 / (1 + np.exp(-z / t)), EPS, 1 - EPS)
        return -np.mean(y_val * np.log(q) + (1 - y_val) * np.log(1 - q))
    return minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x


def apply_temperature(p, t):
    return 1 / (1 + np.exp(-log_odds(p) / t))


def ece_top_label(y, p):
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    return sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
               for b in range(N_BINS) if (bins == b).any())


def threshold_for_sensitivity(p, y, target):
    """Largest threshold whose malignant recall on (val) reaches the target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[k - 1]) if k >= 1 else 0.5


def metrics(y, p, thr, mel):
    pred = (p >= thr).astype(int)
    wrong = pred != y
    return {
        "accuracy": (~wrong).mean(),
        "balanced accuracy": 0.5 * (pred[y == 1].mean() + (1 - pred[y == 0]).mean()),
        "AUROC": roc_auc_score(y, p),
        "malignant recall": pred[y == 1].mean(),
        "benign recall": (1 - pred[y == 0]).mean(),
        "melanoma recall": pred[mel].mean(),
        "ECE (top label)": ece_top_label(y, p),
        "Brier": np.mean((p - y) ** 2),
    }


def load_runs(runs):
    frames = {}
    for item in runs:
        name, rel = item.split("=", 1)
        df = pd.read_csv(ROOT / "uncertaintyNet-main" / "output" / rel / "predictions.csv")
        frames[name] = df.set_index("isic_id")
    ids = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    base = next(iter(frames.values())).loc[ids, ["label", "split", "diagnosis_3"]]
    probs = pd.DataFrame({n: f.loc[ids, "p_malignant"] for n, f in frames.items()}, index=ids)
    return base, probs


def main(opts):
    base, probs = load_runs(opts.runs)
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v2.csv").set_index("isic_id")
    base["group_id"] = split.loc[base.index, "group_id"].values
    val, test = base.split == "val", base.split == "test"
    y_val, y_test = base.label[val].values, base.label[test].values
    mel_test = base.diagnosis_3[test].str.contains("melanoma", case=False, na=False).values
    names = list(probs.columns)
    ref = names[0]

    temps = {n: fit_temperature(probs[n][val].values, y_val) for n in names}
    cal = pd.DataFrame({n: apply_temperature(probs[n].values, temps[n]) for n in names}, index=probs.index)

    variants = {  # name -> (probabilities, threshold rule)
        f"{ref} @0.5 (reference)": (probs[ref], "0.5"),
        f"{ref} @val-sens": (probs[ref], "sens"),
        f"{ref} +T @0.5": (cal[ref], "0.5"),
        f"{ref} +T @val-sens": (cal[ref], "sens"),
        "ensemble(all) +T @0.5": (cal[names].mean(axis=1), "0.5"),
        "ensemble(all) +T @val-sens": (cal[names].mean(axis=1), "sens"),
    }
    if opts.subset:
        sub = opts.subset.split(",")
        variants[f"ensemble({'+'.join(sub)}) +T @val-sens"] = (cal[sub].mean(axis=1), "sens")

    groups = base.group_id[test].values
    idx_by_group = pd.Series(np.arange(test.sum())).groupby(groups).apply(np.array)
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([idx_by_group.iloc[i] for i in rng.choice(len(idx_by_group), len(idx_by_group))])
             for _ in range(N_BOOT)]

    resolved = {}
    for name, (p, rule) in variants.items():
        thr = threshold_for_sensitivity(p[val].values, y_val, opts.target_sensitivity) if rule == "sens" else 0.5
        resolved[name] = (p[test].values, thr)
    ref_p, ref_thr = resolved[f"{ref} @0.5 (reference)"]

    rows, report = [], {"temperatures": temps, "target_val_sensitivity": opts.target_sensitivity,
                        "n_test": int(test.sum()), "n_melanoma_test": int(mel_test.sum()), "variants": {}}
    for name, (p, thr) in resolved.items():
        m = metrics(y_test, p, thr, mel_test)
        diffs = {k: [] for k in m}
        for rows_b in boots:
            mb = metrics(y_test[rows_b], p[rows_b], thr, mel_test[rows_b])
            ma = metrics(y_test[rows_b], ref_p[rows_b], ref_thr, mel_test[rows_b])
            for k in m:
                diffs[k].append(mb[k] - ma[k])
        ci = {k: np.nanpercentile(v, [2.5, 97.5]).tolist() for k, v in diffs.items()}
        report["variants"][name] = {"threshold": thr, "test": m, "diff_vs_reference_ci95": ci}
        rows.append((name, thr, m, ci))

    keys = ["AUROC", "balanced accuracy", "malignant recall", "benign recall", "melanoma recall", "ECE (top label)", "Brier"]
    lines = [f"# Operating point, temperature scaling and ensembling (test n = {int(test.sum())}, "
             f"melanoma n = {int(mel_test.sum())})", "",
             f"Runs: {', '.join(names)}. Temperatures (val NLL): " + ", ".join(f"{n} {t:.2f}" for n, t in temps.items())
             + f". `@val-sens`: threshold reaching {opts.target_sensitivity:.0%} malignant recall on val. "
             "Brackets: paired cluster-bootstrap 95% CI of the difference from the reference; * = excludes 0.", "",
             "| variant | thr | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for name, thr, m, ci in rows:
        cells = []
        for k in keys:
            lo, hi = ci[k]
            star = "*" if (lo > 0 or hi < 0) else ""
            cells.append(f"{m[k]:.3f}" + ("" if name.endswith("(reference)") else f" [{lo:+.3f},{hi:+.3f}]{star}"))
        lines.append(f"| {name} | {thr:.2f} | " + " | ".join(cells) + " |")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibrate_ensemble.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "calibrate_ensemble.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_folder (relative to uncertaintyNet-main/output); first = reference")
    ap.add_argument("--subset", default="", help="comma-separated run names for an extra smaller ensemble")
    ap.add_argument("--target_sensitivity", type=float, default=0.90)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
