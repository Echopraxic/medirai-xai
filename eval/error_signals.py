"""
Can explanation-side evidence flag ResNet50 errors that the model's own uncertainty misses? (plan T5.4/T6.5, C6)

The C6 study found that explanations of confidently WRONG predictions look like those of confidently correct ones
(surrogate agreement 92% vs 92%), and only the entropy flag separates them. Here we test label-free signals that
the pipeline already has, mainly on the CONFIDENT predictions (entropy below the validation 75th percentile), where
an extra caveat would add information:

  entropy               ResNet50 entropy of expected (baseline; already flags the uncertain 25%)
  surrogate_doubt       1 - surrogate-tree probability of the ResNet50's predicted class
  ova_doubt             1 - one-vs-all (summed) probability of the predicted class
  gt_tree_doubt         1 - ground-truth binary tree probability of the predicted class
  evidence_against      share of top-3 grouped SHAP magnitude that argues against the prediction
  atypical              log(d_pred / d_other): mean distance to the k nearest TRAINING lesions whose true label is
                        the predicted class, over the same for the other class (> 0: looks more like the other class)
  mask_fail             mask_quality == fail

All thresholds, standardization and the logistic combination are fit on val and reported on test (AUROC with a
bootstrap 95% CI). The proposed caveat rule is chosen on val too.

    python eval/error_signals.py --inputs trees/output_v1.0/explanation_inputs.jsonl \
        --trees trees/output_v1.0 --features features/output/feature_table_v1.0.parquet --out eval/output/error_signals
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
K = 15
SIGNALS = ["surrogate_doubt", "ova_doubt", "gt_tree_doubt", "evidence_against", "atypical", "mask_fail"]


def feature_columns(df):
    return [c for c in df.columns if len(c) > 2 and c[1] == "_" and c[0] in "ABCTS"]


def atypicality(table, ids, pred):
    """log(mean dist to k train lesions labelled pred / mean dist to k train lesions labelled 1-pred)."""
    feats = feature_columns(table)
    train = table[table.split == "train"]
    med = train[feats].median()
    scaler = StandardScaler().fit(train[feats].fillna(med))
    X = pd.DataFrame(scaler.transform(table.set_index("isic_id").loc[ids, feats].fillna(med)), index=ids)
    out = {}
    for lab in (0, 1):
        ref = scaler.transform(train.loc[train.label == lab, feats].fillna(med))
        d, _ = NearestNeighbors(n_neighbors=K).fit(ref).kneighbors(X.values)
        out[lab] = d.mean(1)
    pred = np.asarray(pred)
    d_pred = np.where(pred == 1, out[1], out[0])
    d_other = np.where(pred == 1, out[0], out[1])
    return np.log(d_pred / d_other)


def auroc_ci(y, s, n=2000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], s[i]))
    return round(float(point), 3), [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]


def build(opts):
    rows = []
    with open(opts.inputs, encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            pl = item["llm_input"]
            mags = [abs(r["grouped_shap"]) for r in pl["top_concepts"]]
            against = sum(abs(r["grouped_shap"]) for r in pl["top_concepts"] if r["direction"] == "argues_against")
            rows.append({"isic_id": item["isic_id"], "split": item["split"], "label": item["label"],
                         "pred": int(pl["prediction"] == "malignant"), "mask_quality": item["mask_quality"],
                         "entropy": pl["uncertainty"]["entropy_of_expected"],
                         "evidence_against": against / max(sum(mags), 1e-9)})
    df = pd.DataFrame(rows)
    tp = pd.read_csv(Path(opts.trees) / "tree_predictions.csv").set_index("isic_id").loc[df.isic_id]
    toward = lambda p: np.where(df.pred.values == 1, p, 1 - p)  # noqa: E731  probability of the predicted class
    df["surrogate_doubt"] = 1 - toward(tp.binary_surrogate_p.values)
    df["ova_doubt"] = 1 - toward(tp.ova_p_malignant.values)
    df["gt_tree_doubt"] = 1 - toward(tp.binary_gt_p.values)
    df["mask_fail"] = (df.mask_quality == "fail").astype(int)
    df["atypical"] = atypicality(pd.read_parquet(opts.features), df.isic_id.values, df.pred.values)
    df["wrong"] = (df.pred != df.label).astype(int)
    threshold = float(df.loc[df.split == "val", "entropy"].quantile(0.75))
    df["confident"] = df.entropy <= threshold
    return df, threshold


def main(opts):
    df, threshold = build(opts)
    val, test = df[df.split == "val"], df[df.split == "test"]
    report = {"entropy_threshold_val_p75": threshold,
              "n": {s: {"all": int(len(d)), "wrong": int(d.wrong.sum()), "confident": int(d.confident.sum()),
                        "confident_wrong": int((d.confident & (d.wrong == 1)).sum())} for s, d in (("val", val), ("test", test))}}

    single = {}
    for name in ["entropy"] + SIGNALS:
        single[name] = {"test_all": auroc_ci(test.wrong, test[name]),
                        "test_confident": auroc_ci(test.loc[test.confident, "wrong"], test.loc[test.confident, name]),
                        "val_confident": round(float(roc_auc_score(val.loc[val.confident, "wrong"], val.loc[val.confident, name])), 3)}
    report["single_signal_auroc"] = single

    # logistic combinations, fit on val confident cases only, scored on test confident cases
    combos = {"explanation_side": [s for s in SIGNALS if s != "mask_fail"], "explanation_side+entropy": ["entropy"] + [s for s in SIGNALS if s != "mask_fail"]}
    vc, tc = val[val.confident], test[test.confident]
    report["combined_auroc_test_confident"] = {}
    for name, cols in combos.items():
        sc = StandardScaler().fit(vc[cols])
        lr = LogisticRegression(C=0.5, max_iter=1000).fit(sc.transform(vc[cols]), vc.wrong)
        score = lr.predict_proba(sc.transform(tc[cols]))[:, 1]
        report["combined_auroc_test_confident"][name] = {
            "auroc": auroc_ci(tc.wrong, score), "coef": dict(zip(cols, np.round(lr.coef_[0], 3).tolist()))}

    # caveat rule: best single explanation-side signal on VAL confident cases; flag its top `rate` share of val
    best = max((s for s in SIGNALS if s != "mask_fail"), key=lambda s: single[s]["val_confident"])
    rule = {}
    for rate in (0.1, 0.2, 0.3):
        cut = float(vc[best].quantile(1 - rate))
        flag_v, flag_t = vc[best] > cut, tc[best] > cut
        rule[f"top_{int(rate * 100)}pct"] = {
            "cut": round(cut, 4),
            "test_flag_rate": round(float(flag_t.mean()), 3),
            "test_error_rate_flagged": round(float(tc.wrong[flag_t].mean()), 3) if flag_t.any() else None,
            "test_error_rate_unflagged": round(float(tc.wrong[~flag_t].mean()), 3),
            "test_share_of_confident_errors_caught": round(float(flag_t[tc.wrong == 1].mean()), 3),
            "val_error_rate_flagged": round(float(vc.wrong[flag_v].mean()), 3)}
    report["caveat_rule"] = {"signal": best, "fit_on": "val confident predictions", "options": rule,
                             "base_error_rate_test_confident": round(float(tc.wrong.mean()), 3)}

    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "error_signals_per_image.csv", index=False)
    (out / "error_signals.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(ROOT / "trees/output_v1.0/explanation_inputs.jsonl"))
    ap.add_argument("--trees", default=str(ROOT / "trees/output_v1.0"))
    ap.add_argument("--features", default=str(ROOT / "features/output/feature_table_v1.0.parquet"))
    ap.add_argument("--out", default=str(ROOT / "eval/output/error_signals"))
    main(ap.parse_args())
