"""
C1 surrogate experiment (plan decision 2026-11-04, fixed before the run). Final model kdv8f_mel s3, features v1.1.

Arms (2 x 2 x 2): target hard (ResNet decision; XGBoost classifier, current) / soft (regression on logit p_malignant,
malignant if predicted p >= 0.5); training rows without mask-fail (current) / with; features all / without the ones that
have no clinical concept in phrase bank v3. Same tuning grid as trees/train_trees.py, early stopping on validation.
Selection on split-v2 val (all rows): score = min(agreement among true benign, among true malignant). Adopt the best arm
only if score >= baseline + 0.01 and overall val agreement >= baseline - 0.005; test is reported once (best and baseline).
Melanoma curve (measurement only): baseline arm with 0/25/50/75% of training melanomas dropped, 5 subsample seeds.

    python eval/surrogate_c1.py   -> eval/output/surrogate_c1/report.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from trees.train_trees import GRID, SEED, agreement, feature_columns

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "uncertaintyNet-main/output/narval_stageb_kdv8f_mel_s3_20261030/run/predictions.csv"
FEATURES = ROOT / "features/output/feature_table_v1.1.parquet"
BANK = ROOT / "concepts/output/phrase_bank_v3.csv"
OUT = ROOT / "eval/output/surrogate_c1"


def concept_less(feats):
    pb = pd.read_csv(BANK, encoding="utf-8")
    named = {f for f, c in zip(pb.feature_id, pb.intended_clinical_concept.fillna("")) if len(str(c).strip()) >= 3}
    return [f for f in feats if f not in named]


def load():
    table = pd.read_parquet(FEATURES)
    preds = pd.read_csv(RUN)[["isic_id", "pred", "p_malignant"]].rename(columns={"pred": "resnet_pred", "p_malignant": "resnet_p"})
    df = table.merge(preds, on="isic_id")
    df["label"] = df.label.astype(int)
    return df[df.extraction_error.fillna("") == ""].reset_index(drop=True)


def fit(tr, va, feats, soft, seed=SEED):
    """Grid on validation loss with early stopping; returns a function rows -> malignant call (0/1)."""
    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        if soft:
            y_tr = np.log(np.clip(tr.resnet_p, 1e-4, 1 - 1e-4) / (1 - np.clip(tr.resnet_p, 1e-4, 1 - 1e-4)))
            y_va = np.log(np.clip(va.resnet_p, 1e-4, 1 - 1e-4) / (1 - np.clip(va.resnet_p, 1e-4, 1 - 1e-4)))
            m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, early_stopping_rounds=50, eval_metric="rmse",
                                 tree_method="hist", random_state=seed, n_jobs=6, **params)
        else:
            y_tr, y_va = tr.resnet_pred, va.resnet_pred
            pos = y_tr.mean()
            m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, early_stopping_rounds=50, eval_metric="logloss",
                                  scale_pos_weight=(1 - pos) / pos, tree_method="hist", random_state=seed, n_jobs=6, **params)
        m.fit(tr[feats], y_tr, eval_set=[(va[feats], y_va)], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (m.best_score, m)
    m = best[1]
    return (lambda d: (m.predict(d[feats]) >= 0).astype(int)) if soft else (lambda d: (m.predict_proba(d[feats])[:, 1] >= 0.5).astype(int))


def scores(d, call):
    c = call(d)
    by = {int(k): agreement(c[(d.label == k).to_numpy()], g.resnet_pred) for k, g in d.groupby("label")}
    mel = (d.ova_class == "Melanoma").to_numpy()
    return {"overall": agreement(c, d.resnet_pred), "benign": by[0], "malignant": by[1], "min_class": min(by.values()),
            "resnet_benign_calls": agreement(c[(d.resnet_pred == 0).to_numpy()], d.resnet_pred[d.resnet_pred == 0]),
            "resnet_malignant_calls": agreement(c[(d.resnet_pred == 1).to_numpy()], d.resnet_pred[d.resnet_pred == 1]),
            "melanoma": agreement(c[mel], d.resnet_pred[mel]) if mel.any() else None, "n_melanoma": int(mel.sum())}


def main():
    df = load()
    feats = feature_columns(df)
    drop = concept_less(feats)
    val, test = df[df.split == "val"], df[df.split == "test"]
    res = {"n_features": len(feats), "concept_less_features": drop, "arms": {}}
    calls = {}
    for soft, with_fail, reduced in itertools.product((False, True), (False, True), (False, True)):
        name = f"{'soft' if soft else 'hard'}|{'with_fail' if with_fail else 'no_fail'}|{'reduced' if reduced else 'all'}"
        f = [x for x in feats if x not in drop] if reduced else feats
        keep = (df.mask_quality != "fail") | with_fail
        call = fit(df[(df.split == "train") & keep], df[(df.split == "val") & keep], f, soft)
        calls[name] = call
        res["arms"][name] = {"n_features": len(f), "n_train": int(((df.split == "train") & keep).sum()), "val": scores(val, call)}
        print(name, {k: round(v, 4) for k, v in res["arms"][name]["val"].items() if isinstance(v, float)}, flush=True)
    base = "hard|no_fail|all"
    b = res["arms"][base]["val"]
    best = max(res["arms"], key=lambda k: (res["arms"][k]["val"]["min_class"], res["arms"][k]["val"]["overall"]))
    v = res["arms"][best]["val"]
    adopt = best != base and v["min_class"] >= b["min_class"] + 0.01 and v["overall"] >= b["overall"] - 0.005
    res["selection"] = {"baseline": base, "best": best, "adopt": bool(adopt),
                        "test": {k: scores(test, calls[k]) for k in dict.fromkeys([base, best])}}
    # melanoma curve (measurement only)
    curve = {}
    tr_all = df[(df.split == "train") & (df.mask_quality != "fail")]
    va_nf = df[(df.split == "val") & (df.mask_quality != "fail")]
    mel_idx = tr_all.index[tr_all.ova_class == "Melanoma"]
    for frac in (0.0, 0.25, 0.5, 0.75):
        vals = []
        for s in range(5 if frac else 1):
            rng = np.random.default_rng(s)
            dropped = rng.choice(mel_idx, int(frac * len(mel_idx)), replace=False)
            call = fit(tr_all.drop(index=dropped), va_nf, feats, soft=False)
            vals.append(scores(val, call)["melanoma"])
        curve[f"drop_{frac:.2f}"] = {"n_train_melanoma": int(len(mel_idx) * (1 - frac)), "val_melanoma_agreement_mean": float(np.mean(vals)),
                                     "sd": float(np.std(vals))}
        print("melanoma curve", frac, curve[f"drop_{frac:.2f}"], flush=True)
    res["melanoma_curve"] = curve
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["selection"], indent=1))


if __name__ == "__main__":
    main()
