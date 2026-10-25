"""
S1-9 misclassified and deferred lists for the final candidate on the locked external sets.

Reads the locked-set predictions S2 sent (eval/output/stage_c/locked_final/locked_final_<set>_predictions.csv)
and the internal val probabilities (eval/output/stage_c/final/scores.npz). No model is run and nothing is selected:
the locked sets are only described, so this does not count as a new look.

Deferred = entropy of expected above the internal v2-val 75th percentile, the same rule as locked_final.json
("deferred share (internal val p75 entropy)"); the script checks it reproduces those shares.

Each image gets one stratum:
  confident_wrong   error the entropy rule does not catch (the main target for TreeSHAP)
  deferred_wrong    error the entropy rule catches
  deferred_right    correct but sent to a clinician anyway (cost of deferral)
  confident_right   the rest

Outputs (in --out; DDI/DDI-2 rows fall under Stanford AIMI research-use terms, keep local, do not commit):
  <set>_errors.csv     misclassified images (FN/FP), sorted by confidence of the wrong answer
  <set>_deferred.csv   deferred images, right or wrong
  all_strata.csv       every image of every set with its stratum
  summary.md           counts per set, per skin-tone group and per diagnosis

    python eval/s1_error_lists.py --results ~/Downloads/s1_results_2026-10-23 --out eval/output/s1_error_lists
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SETS = ["isic_overview", "mcr_sl", "ddi", "ddi2"]


def entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def stratum(row):
    wrong = row.pred != row.label
    if row.deferred:
        return "deferred_wrong" if wrong else "deferred_right"
    return "confident_wrong" if wrong else "confident_right"


def table(df, by):
    g = df.groupby(by)
    out = pd.DataFrame({
        "n": g.size(),
        "malignant": g.label.sum(),
        "error_rate": g.wrong.mean(),
        "FN": g.apply(lambda x: ((x.label == 1) & (x.pred == 0)).sum()),
        "FP": g.apply(lambda x: ((x.label == 0) & (x.pred == 1)).sum()),
        "deferred_share": g.deferred.mean(),
        "confident_wrong": g.apply(lambda x: (x.stratum == "confident_wrong").sum()),
    })
    return out.round(3)


def main(opts):
    res = Path(opts.results).expanduser()
    locked = json.loads((res / "eval/output/stage_c/locked_final/locked_final.json").read_text())
    val_p = np.load(res / "eval/output/stage_c/final/scores.npz")["v2_val_prob"]
    threshold = float(np.quantile(entropy(val_p), 0.75))

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, lines = [], [f"# S1-9 error and deferral lists ({locked['run']})", "",
                         f"Deferred = entropy of expected > {threshold:.4f} (internal v2 val 75th percentile). "
                         f"Decision threshold {locked['threshold']:.4f} (standard mode).", ""]

    for name in SETS:
        df = pd.read_csv(res / f"eval/output/stage_c/locked_final/locked_final_{name}_predictions.csv")
        df["set"] = name
        df["deferred"] = df.entropy_of_expected > threshold
        expected = locked["sets"][name]["deferred share (internal val p75 entropy)"]
        assert abs(df.deferred.mean() - expected) < 1e-9, f"{name}: deferral share does not match locked_final.json"
        df["wrong"] = df.pred != df.label
        df["error_type"] = np.select([(df.label == 1) & (df.pred == 0), (df.label == 0) & (df.pred == 1)],
                                     ["FN", "FP"], "")
        df["melanoma"] = df.diagnosis.str.contains("melanoma", case=False, na=False)
        df["wrong_confidence"] = np.where(df.pred == 1, df.p_malignant, 1 - df.p_malignant)
        df["stratum"] = df.apply(stratum, axis=1)
        if "tone" not in df:
            df["tone"] = "not recorded"
        frames.append(df)

        df[df.wrong].sort_values("wrong_confidence", ascending=False).to_csv(out / f"{name}_errors.csv", index=False)
        df[df.deferred].sort_values("entropy_of_expected", ascending=False).to_csv(out / f"{name}_deferred.csv",
                                                                                     index=False)
        s = locked["sets"][name]
        lines += [f"## {name}", "",
                  f"n = {len(df)}, errors = {df.wrong.sum()} (FN {(df.error_type == 'FN').sum()}, "
                  f"FP {(df.error_type == 'FP').sum()}), melanomas missed = {(df.melanoma & (df.pred == 0)).sum()}"
                  f" of {df.melanoma.sum()}; AUROC {s['AUROC']['value']:.3f}, "
                  f"entropy AUROC for errors {s['entropy AUROC for errors']:.3f}.", "",
                  df.stratum.value_counts().rename("n").to_frame().to_markdown(), ""]
        if (df.tone != "not recorded").any():
            lines += ["By skin tone:", "", table(df, "tone").to_markdown(), ""]
        diag = table(df, "diagnosis")
        diag = diag[diag.n >= opts.min_n].sort_values("error_rate", ascending=False).head(opts.top)
        lines += [f"Diagnoses with n >= {opts.min_n}, highest error rate first:", "", diag.to_markdown(), ""]

    allx = pd.concat(frames, ignore_index=True)
    allx.to_csv(out / "all_strata.csv", index=False)
    lines += ["## All sets", "", table(allx, "set").to_markdown(), "",
              "Error rate in deferred vs retained images:", "",
              allx.groupby(["set", "deferred"]).wrong.mean().unstack().round(3).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="unzipped s1_results folder")
    ap.add_argument("--out", default="eval/output/s1_error_lists")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--top", type=int, default=12)
    main(ap.parse_args())
