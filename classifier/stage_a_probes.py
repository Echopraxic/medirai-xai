"""
Stage A3 of the classifier sprint: do foundation-model representations generalize to phone photos, and how much does
MIDAS training data help? Linear probes on frozen embeddings, scored on the SELECTION data only (split v2 val + MIDAS
dev_val phone photos; pre-registration 2026-10-21).

Per backbone (embeddings from classifier/embed_probe.py: <model>.npz for split v2, <model>__midas.npz for MIDAS):
  v2_only        L2 logistic regression fit on split v2 train
  v2+midas       fit on split v2 train + MIDAS train (all modalities' phone photos only), preview of stage B5
C is chosen on the selection score S = mean(internal val AUROC, MIDAS dev_val AUROC), the same rule as the prereg.

    python classifier/stage_a_probes.py --models dinov2 monet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "classifier/output/embeddings"
C_GRID = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def load(model):
    z = np.load(EMB / f"{model}.npz", allow_pickle=True)
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv").set_index("isic_id")
    e = pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"])
    e = e.loc[e.index.intersection(v2.index[v2.split.isin(["train", "val"])])]
    a = pd.DataFrame({"split": v2.loc[e.index, "split"].map({"train": "v2_train", "val": "internal_val"}),
                      "label": v2.loc[e.index, "label"].astype(int)}, index=e.index)
    z = np.load(EMB / f"{model}__midas.npz", allow_pickle=True)
    m = pd.read_csv(ROOT / "splits/midas_v1.csv").set_index("image_id")
    em = pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"])
    m = m.loc[em.index]
    keep = m.modality.str.startswith("phone") & m.split.isin(["train", "dev_val"])
    em, m = em[keep.values], m[keep.values]
    b = pd.DataFrame({"split": m.split.map({"train": "midas_train", "dev_val": "midas_dev_val"}),
                      "label": m.label.astype(int)}, index=m.index)
    return pd.concat([e, em]), pd.concat([a, b])


def run(model):
    X, meta = load(model)
    rows = []
    for variant, train_splits in (("v2_only", ["v2_train"]), ("v2+midas", ["v2_train", "midas_train"])):
        tr = meta.split.isin(train_splits).values
        scaler = StandardScaler().fit(X.values[tr])
        Xs = scaler.transform(X.values)
        best = None
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(Xs[tr], meta.label.values[tr])
            p = clf.predict_proba(Xs)[:, 1]
            r = {f"{s}_auroc": float(roc_auc_score(meta.label[meta.split == s], p[(meta.split == s).values]))
                 for s in ("internal_val", "midas_dev_val")}
            r["S"] = (r["internal_val_auroc"] + r["midas_dev_val_auroc"]) / 2
            if best is None or r["S"] > best["S"]:
                best = {"model": model, "variant": variant, "C": c, **r}
        rows.append(best)
        print(json.dumps(best), flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dinov2", "monet"])
    opts = ap.parse_args()
    df = pd.DataFrame([r for m in opts.models for r in run(m)]).round(4)
    out = ROOT / "eval/output/stage_a/probes"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "probes.csv", index=False)
    (out / "summary.md").write_text("# Stage A3: linear probes on frozen embeddings (selection data only)\n\n"
                                    + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
