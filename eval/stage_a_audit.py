"""
Stage A2 of the classifier sprint: which image properties predict the ResNet's errors on phone photos? MONET concept
scores (concepts/monet_concepts.py, the auditing use of Kim et al. 2024) and MIDAS metadata (camera distance,
Fitzpatrick type, lesion length in mm) against errors on MIDAS dev_val (selection data; no locked set is read).

Errors are split by true class, because concepts correlate with the label itself: for malignant lesions an error is a
missed cancer (false negative), for benign lesions a false alarm. Per property: AUROC of the property for "error vs
correct" within the class (0.5 = unrelated; > 0.5 = higher values go with errors), with a 95% patient-bootstrap CI.

    python eval/stage_a_audit.py --run rorqual_v2_lr3e-4_20261014
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
QA_ONLY = ("M_artifact_",)


def boot_auc(y, x, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in g}
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if len(set(y[take])) == 2:
            vals.append(roc_auc_score(y[take], x[take]))
    return np.percentile(vals, [2.5, 97.5])


def main(opts):
    pred = pd.read_csv(ROOT / "eval/output/external/midas_dev_val/predictions.csv")
    p_col = f"{opts.run}__p_malignant"
    pred["error"] = ((pred[p_col] >= 0.5).astype(int) != pred.label).astype(int)
    split = pd.read_csv(ROOT / "splits/midas_v1.csv")[["image_id", "patient_id", "modality", "midas_fitzpatrick",
                                                        "length__mm_"]]
    conc = pd.read_parquet(ROOT / "concepts/output/monet_concepts_v1__midas.parquet").rename(columns={"isic_id": "image_id"})
    df = pred.merge(split, on="image_id").merge(conc, on="image_id")
    df["distance_1ft"] = (df.modality == "phone_1ft").astype(float)
    fst = df.midas_fitzpatrick.astype(str).str.extract(r"^(vi|v|iv|iii|ii|i)\b")[0].map(
        {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6})
    df["fitzpatrick"] = fst.astype(float)
    df["lesion_length_mm"] = pd.to_numeric(df.length__mm_, errors="coerce")
    props = [c for c in conc.columns if c.startswith("M_")] + ["distance_1ft", "fitzpatrick", "lesion_length_mm"]
    rows = []
    for cls, name in ((1, "malignant (missed cancer)"), (0, "benign (false alarm)")):
        d = df[df.label == cls]
        for p in props:
            ok = d[p].notna().to_numpy()
            y, x, g = d.error.to_numpy()[ok], d[p].to_numpy()[ok], d.patient_id.to_numpy()[ok]
            if len(set(y)) < 2:
                continue
            auc = roc_auc_score(y, x)
            lo, hi = boot_auc(y, x, g)
            rows.append({"class": name, "property": p, "qa_only": p.startswith(QA_ONLY), "n": int(ok.sum()),
                         "error_rate": round(float(y.mean()), 3), "auroc_for_error": round(auc, 3),
                         "ci95": f"[{lo:.2f}, {hi:.2f}]", "ci_excludes_0.5": bool(lo > 0.5 or hi < 0.5)})
    res = pd.DataFrame(rows)
    res["dist"] = (res.auroc_for_error - 0.5).abs()
    res = res.sort_values(["class", "dist"], ascending=[True, False]).drop(columns="dist")
    out = ROOT / "eval/output/stage_a/audit"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"audit_{opts.run}.csv", index=False)
    (out / f"audit_{opts.run}.md").write_text(
        f"# Stage A2: what predicts errors on MIDAS dev_val phone photos ({opts.run}, threshold 0.5)\n\n"
        + res.to_markdown(index=False) + "\n", encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rorqual_v2_lr3e-4_20261014")
    main(ap.parse_args())
