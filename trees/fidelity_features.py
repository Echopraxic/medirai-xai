"""
C1 surrogate fidelity vs feature set: does the tree agree more with ResNet50 when it sees more than the 42 handcrafted
features? Training recipe fixed to the surrogate sweep's best (ResNet hard labels, fail-mask rows kept, no class
reweighting, decision threshold chosen on val); only the feature set changes. Selection and thresholds use val only.

Feature sets:
  hand            the 42 handcrafted features (feature table v1.0)
  monet           12 named MONET concept scores (concepts/monet_concepts.py; artifact concepts excluded)
  hand+monet      both: every input still has a clinical name
  hand+emb16      + 16 PCs of the explained ResNet's own penultimate embedding (fit on train). Upper bound with
                  unnamed inputs: the tree partly copies the model, so it is a ceiling, not a candidate
  hand+monet+emb16

Test agreement is reported overall, per true class (C1's definition) and per stratum (mask quality, deferral).

    python trees/fidelity_features.py --run rorqual_v2_lr3e-4_20261014/resnet50_det/run \
        --out trees/output_v1.0_detv2/fidelity_features
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trees"))

from surrogate_sweep import best_threshold, fidelity, fit, proba  # noqa: E402
from train_trees import feature_columns  # noqa: E402

MONET = ROOT / "concepts/output/monet_concepts_v1.parquet"
MONET_QA = ("M_artifact_",)


def main(opts):
    run = ROOT / "uncertaintyNet-main/output" / opts.run
    df = pd.read_parquet(ROOT / opts.features)
    hand = feature_columns(df)
    preds = pd.read_csv(run / "predictions.csv")[["isic_id", "pred", "p_malignant", "entropy_of_expected"]].rename(
        columns={"pred": "resnet_pred", "p_malignant": "resnet_p"})
    df = df.merge(preds, on="isic_id").merge(pd.read_parquet(MONET), on="isic_id")
    monet = [c for c in df.columns if c.startswith("M_") and not c.startswith(MONET_QA)]
    df = df[df.extraction_error.fillna("") == ""].reset_index(drop=True)
    df["label"] = df.label.astype(int)
    emb_cols = []
    emb_file = run / "embeddings.npz"
    if emb_file.exists():
        z = np.load(emb_file, allow_pickle=True)
        emb = pd.DataFrame(z["emb"].astype(np.float32), index=z["isic_id"]).loc[df.isic_id].to_numpy()
        pca = PCA(16, random_state=0).fit(emb[(df.split == "train").to_numpy()])
        emb_cols = [f"E_pc{i}" for i in range(16)]
        df[emb_cols] = pca.transform(emb)
    thr_eoe = df.loc[df.split == "val", "entropy_of_expected"].quantile(0.75)
    df["deferred"] = df.entropy_of_expected > thr_eoe
    sets = {"hand": hand, "monet": monet, "hand+monet": hand + monet}
    if emb_cols:
        sets.update({"hand+emb16": hand + emb_cols, "hand+monet+emb16": hand + monet + emb_cols})
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    rows = {}
    for name, cols in sets.items():
        m, params = fit(tr[cols], tr.resnet_pred, va[cols], va.resnet_pred, soft=False, balanced=False)
        thr = best_threshold(proba(m, va[cols]), va.resnet_pred.values)
        p_te = proba(m, te[cols]) >= thr
        strata = {}
        for col, groups in (("mask_quality", ["ok", "review", "fail"]), ("deferred", [False, True])):
            for g in groups:
                s = (te[col] == g).to_numpy()
                strata[f"{col}={g}"] = {"n": int(s.sum()), "agreement": float((p_te[s] == te.resnet_pred.values[s]).mean())}
        rows[name] = {"n_features": len(cols), "threshold": thr, "params": params,
                      "val": fidelity(proba(m, va[cols]) >= thr, va.resnet_pred),
                      "test": fidelity(p_te, te.resnet_pred, te.label), "test_strata": strata}
        t = rows[name]["test"]
        print(f"{name:18s} n={len(cols):3d} test {t['overall']:.3f}  per true class benign {t['by_true_label'][0]:.3f} "
              f"malignant {t['by_true_label'][1]:.3f} | val {rows[name]['val']['overall']:.3f}", flush=True)
    out = ROOT / opts.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity_features.json").write_text(json.dumps({"run": opts.run, "features": opts.features,
                                                            "monet": str(MONET.relative_to(ROOT)), "sets": rows}, indent=1))
    lines = [f"# C1 fidelity by feature set ({opts.run})", "",
             "| feature set | n | test agreement | benign (true) | malignant (true) | mask ok | mask fail | not deferred | deferred |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        t, s = r["test"], r["test_strata"]
        lines.append(f"| {name} | {r['n_features']} | {t['overall']:.3f} | {t['by_true_label'][0]:.3f} | "
                     f"{t['by_true_label'][1]:.3f} | {s['mask_quality=ok']['agreement']:.3f} | "
                     f"{s['mask_quality=fail']['agreement']:.3f} | {s['deferred=False']['agreement']:.3f} | "
                     f"{s['deferred=True']['agreement']:.3f} |")
    (out / "fidelity_features.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder relative to uncertaintyNet-main/output")
    ap.add_argument("--features", default="features/output/feature_table_v1.0.parquet")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
