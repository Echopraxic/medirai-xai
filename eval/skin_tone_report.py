"""
Classifier performance by skin tone on the split-v2 test set (WS9 fairness check, no new data needed).

Two tone scales exist for our images and are kept separate in the table:
  - MILK10k `skin_tone_class` (0 = darkest ... 5 = lightest; ISIC MILK10k documentation), for every MILK image;
  - ISIC `fitzpatrick_skin_type` (I-VI), recorded for some UFES images (and MSKCC, which v2 excludes).
For a single report column they are mapped to a coarse group: darker = MILK 0-2 or FST V-VI, medium = MILK 3 or
FST III-IV, lighter = MILK 4-5 or FST I-II. The two scales are not equivalent; the coarse group is for screening,
and per-scale counts are printed so a gap can be traced to its source.

Metrics per group: n, malignant share, accuracy, balanced accuracy, AUROC, malignant and benign recall, each with a
95% percentile interval from a cluster bootstrap over lesion/patient groups (`group_id`) inside the group.

    python eval/skin_tone_report.py --runs det=rorqual_v2_lr3e-4_20261014/resnet50_det/run \
        v3det=narval_v3_medir_v3_lr3e-4_20261015/resnet50_det/run --out eval/output/skin_tone
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits/isic_clinical_v2.csv"
MILK_TONES = ROOT / "data/milk10k/supplements/training_input.csv"
ISIC_TONES = ROOT / "splits/sources/isic_clinical_fitzpatrick.csv"
RUNS_ROOT = ROOT / "uncertaintyNet-main/output"
GROUPS = ["darker", "medium", "lighter", "unknown"]


def coarse_group(milk, fst):
    if pd.notna(milk):
        return "darker" if milk <= 2 else "medium" if milk == 3 else "lighter"
    if isinstance(fst, str) and fst:
        return "darker" if fst in ("V", "VI") else "medium" if fst in ("III", "IV") else "lighter"
    return "unknown"


def tone_table(split, milk, fitz):
    t = split[["isic_id", "split", "label", "group_id", "source"]].merge(
        milk[["isic_id", "skin_tone_class"]].rename(columns={"skin_tone_class": "milk_tone"}), on="isic_id", how="left")
    t = t.merge(fitz[["isic_id", "fitzpatrick_skin_type"]].rename(columns={"fitzpatrick_skin_type": "fst"}),
                on="isic_id", how="left")
    t["tone_group"] = [coarse_group(m, f) for m, f in zip(t.milk_tone, t.fst)]
    return t


def fetch_fitzpatrick():
    sys.path.insert(0, str(ROOT / "splits"))
    from fetch_isic_metadata import fetch
    rows = fetch('image_type:"clinical: close-up"')
    df = pd.DataFrame(rows)[["isic_id", "fitzpatrick_skin_type"]]
    ISIC_TONES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ISIC_TONES, index=False)
    return df


def metrics(y, p, pred):
    out = {"n": int(len(y)), "malignant_share": float(y.mean()) if len(y) else np.nan,
           "accuracy": float((pred == y).mean()) if len(y) else np.nan}
    rec1 = float((pred[y == 1] == 1).mean()) if (y == 1).any() else np.nan
    rec0 = float((pred[y == 0] == 0).mean()) if (y == 0).any() else np.nan
    out.update(malignant_recall=rec1, benign_recall=rec0, balanced_accuracy=np.nanmean([rec0, rec1]),
               auroc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else np.nan)
    return out


def bootstrap(df, n_boot=1000, seed=0):
    """Percentile intervals, resampling lesion/patient groups within the stratum."""
    rng = np.random.default_rng(seed)
    groups = df.group_id.unique()
    idx = {g: np.flatnonzero(df.group_id.values == g) for g in groups}
    keys = ["accuracy", "balanced_accuracy", "auroc", "malignant_recall", "benign_recall"]
    draws = {k: [] for k in keys}
    y, p, pred = df.label.values, df.p_malignant.values, df.pred.values
    for _ in range(n_boot):
        rows = np.concatenate([idx[g] for g in rng.choice(groups, len(groups), replace=True)])
        m = metrics(y[rows], p[rows], pred[rows])
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def main(opts):
    split = pd.read_csv(SPLIT)
    split = split[split.split.isin(["train", "val", "test"])]
    milk = pd.read_csv(MILK_TONES)
    fitz = pd.read_csv(ISIC_TONES) if ISIC_TONES.exists() and not opts.refetch else fetch_fitzpatrick()
    tones = tone_table(split, milk, fitz)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    tones.to_csv(out / "skin_tone_table.csv", index=False)

    lines = ["# Classifier performance by skin tone (split v2)", "",
             "MILK10k tone 0 = darkest ... 5 = lightest; darker = MILK 0-2 or FST V-VI, medium = MILK 3 or FST III-IV, "
             "lighter = MILK 4-5 or FST I-II. 95% cluster-bootstrap intervals over lesion/patient groups.", "",
             "## Coverage (all splits)", "",
             pd.crosstab([tones.split], tones.tone_group).reindex(columns=GROUPS, fill_value=0).to_markdown(), "",
             "Test images by scale: MILK class " + json.dumps(
                 tones[tones.split == "test"].milk_tone.value_counts().sort_index().astype(int)
                 .rename(lambda k: str(int(k))).to_dict())
             + ", Fitzpatrick " + json.dumps(tones[tones.split == "test"].fst.value_counts().sort_index().to_dict()), ""]
    results = {}
    for spec in opts.runs:
        name, folder = spec.split("=", 1)
        pr = pd.read_csv(RUNS_ROOT / folder / "predictions.csv")
        test = pr[pr.split == "test"][["isic_id", "label", "pred", "p_malignant"]].merge(
            tones[["isic_id", "group_id", "tone_group"]], on="isic_id")
        lines += [f"## {name}", "", "| tone | n | malignant share | accuracy | balanced acc | AUROC | malignant recall | "
                  "benign recall |", "|---|---|---|---|---|---|---|---|"]
        results[name] = {}
        for g in GROUPS + ["all"]:
            d = test if g == "all" else test[test.tone_group == g]
            if len(d) == 0:
                continue
            m = metrics(d.label.values, d.p_malignant.values, d.pred.values)
            ci = bootstrap(d, opts.n_boot) if len(d) >= 20 else {}
            m["ci"] = ci
            results[name][g] = m

            def cell(k):
                v = m[k]
                if np.isnan(v):
                    return "n/a"
                return f"{v:.3f}" + (f" [{ci[k][0]:.2f}, {ci[k][1]:.2f}]" if k in ci else "")
            lines.append(f"| {g} | {m['n']} | {m['malignant_share']:.2f} | {cell('accuracy')} | {cell('balanced_accuracy')} "
                         f"| {cell('auroc')} | {cell('malignant_recall')} | {cell('benign_recall')} |")
        lines.append("")
    (out / "skin_tone_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "skin_tone_report.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_folder (relative to uncertaintyNet-main/output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--refetch", action="store_true", help="re-download the ISIC Fitzpatrick column")
    main(ap.parse_args())
