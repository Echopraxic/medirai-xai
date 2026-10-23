"""
Per-domain operating policy for the final model (decision 2026-10-22). The standard threshold (sensitivity 0.90) was
fit on close-up photos; on MIDAS phone photos sensitivity fell to 0.66. Here each domain gets its own calibration
and conformal threshold, fit on THAT domain's validation data, and is evaluated on the matching held-out test split.

  domain      "close-up" vs "phone": logistic regression on frozen DINOv2-L features, trained on split v2 train vs MIDAS
              train phone photos (never on val/test), accuracy reported on val and test
  decision p  mean calibrated probability over a lesion's photos (adopted lesion averaging), then per-domain temperature
              (fit on the domain's validation NLL; applied on top of the global temperature)
  threshold   split-conformal: with n_pos validation malignant scores sorted ascending, t = score[k-1], k = floor(alpha*
              (n_pos+1)), alpha = 0.10; under exchangeability sensitivity >= 0.90 in expectation. Reported next to the
              observed test sensitivity with its 95% binomial interval
  compare     (a) global threshold (current), (b) per-domain threshold with the domain known, (c) per-domain threshold with
              the domain predicted. Reported per test split: sensitivity (+ interval), specificity, ECE

Selection data for fitting (split v2 val; MIDAS dev_val); test data for the one-time report (split v2 test; MIDAS dev_test).
No locked set is read.

    python eval/domain_policy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from final_model_report import ece  # noqa: E402

ALPHA = 0.10


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def binom_ci(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def frames():
    s = np.load(ROOT / "eval/output/stage_c/final/scores.npz")
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone")]
    out = {}
    for sp, name in (("val", "v2_val"), ("test", "v2_test")):
        x = v2[v2.split == sp]
        out[("close-up", sp)] = pd.DataFrame({"id": x.isic_id.to_numpy(), "label": x.label.astype(int).to_numpy(),
                                              "lesion": x.lesion_id.fillna(x.isic_id).to_numpy(), "p": s[f"{name}_prob"]})
    for sp, name, key in (("val", "midas_dev_val", "dev_val"), ("test", "midas_dev_test", "dev_test")):
        x = mid[mid.split == key]
        out[("phone", sp)] = pd.DataFrame({"id": x.image_id.to_numpy(), "label": x.label.astype(int).to_numpy(),
                                           "lesion": x.lesion_id.to_numpy(), "p": s[f"{name}_prob"]})
    for d in out.values():
        d["p_dec"] = d.groupby("lesion").p.transform("mean")
    return out


def domain_classifier():
    emb = {}
    for f in ("dinov2.npz", "dinov2__midas.npz"):
        z = np.load(ROOT / "classifier/output/embeddings" / f, allow_pickle=True)
        emb.update({k: v for k, v in zip(z["isic_id"], z["emb"].astype(np.float32))})
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone")]
    tr = pd.concat([pd.DataFrame({"id": v2[v2.split == "train"].isic_id, "y": 0}),
                    pd.DataFrame({"id": mid[mid.split == "train"].image_id, "y": 1})])
    X = np.stack([emb[i] for i in tr.id])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(X), tr.y)
    return lambda ids: clf.predict_proba(sc.transform(np.stack([emb[i] for i in ids])))[:, 1]


def fit_temperature(p, y):
    z = logit(p)

    def nll(T):
        m = z / T
        return float(np.mean(np.logaddexp(0, -m) * y + np.logaddexp(0, m) * (1 - y)))
    return float(minimize_scalar(nll, bounds=(0.3, 5), method="bounded").x)


def conformal_threshold(p, y, alpha=ALPHA):
    pos = np.sort(p[y == 1])
    k = max(int(np.floor(alpha * (len(pos) + 1))), 1)
    return float(pos[k - 1])


def metrics(y, p, t):
    pred = p >= t
    npos = int((y == 1).sum())
    sens_k = int(pred[y == 1].sum())
    return {"sensitivity": sens_k / npos, "sensitivity_ci95": binom_ci(sens_k, npos), "specificity": float((~pred[y == 0]).mean()),
            "ece": ece(p, y), "flagged_share": float(pred.mean())}


def main():
    fr = frames()
    pdom = domain_classifier()
    t_global = json.loads((ROOT / "eval/output/stage_c/final/final_model_report.json").read_text())["threshold_standard"]
    res = {"alpha": ALPHA, "global_threshold": t_global, "domains": {}, "domain_classifier": {}}
    for sp in ("val", "test"):
        ids = np.concatenate([fr[(d, sp)].id.to_numpy() for d in ("close-up", "phone")])
        truth = np.concatenate([np.zeros(len(fr[("close-up", sp)])), np.ones(len(fr[("phone", sp)]))])
        pr = pdom(ids)
        res["domain_classifier"][sp] = {"accuracy": float(((pr >= 0.5) == truth).mean()), "auroc": float(roc_auc_score(truth, pr))}
    pol = {}
    for dom in ("close-up", "phone"):
        v = fr[(dom, "val")]
        T = fit_temperature(v.p_dec.to_numpy(), v.label.to_numpy())
        pv = 1 / (1 + np.exp(-logit(v.p_dec.to_numpy()) / T))
        pol[dom] = {"temperature": T, "threshold": conformal_threshold(pv, v.label.to_numpy())}
    res["policy"] = pol
    for dom in ("close-up", "phone"):
        te = fr[(dom, "test")]
        y, p = te.label.to_numpy(), te.p_dec.to_numpy()
        own = 1 / (1 + np.exp(-logit(p) / pol[dom]["temperature"]))
        pred_dom = pdom(te.id.to_numpy())
        routed_p = np.where(pred_dom >= 0.5, 1 / (1 + np.exp(-logit(p) / pol["phone"]["temperature"])),
                            1 / (1 + np.exp(-logit(p) / pol["close-up"]["temperature"])))
        routed_t = np.where(pred_dom >= 0.5, pol["phone"]["threshold"], pol["close-up"]["threshold"])
        pred = routed_p >= routed_t
        npos = int((y == 1).sum())
        res["domains"][dom] = {
            "n_test": int(len(te)), "n_malignant": npos,
            "global_threshold": metrics(y, p, t_global),
            "per_domain_known": metrics(y, own, pol[dom]["threshold"]),
            "per_domain_predicted_domain": {"sensitivity": float(pred[y == 1].mean()),
                                            "sensitivity_ci95": binom_ci(int(pred[y == 1].sum()), npos),
                                            "specificity": float((~pred[y == 0]).mean()), "ece": ece(routed_p, y),
                                            "domain_misrouted_share": float(((pred_dom >= 0.5) != (dom == "phone")).mean())}}
    out = ROOT / "eval/output/stage_c/domain_policy.json"
    out.write_text(json.dumps(res, indent=1))
    print("domain classifier", {k: {a: round(b, 3) for a, b in v.items()} for k, v in res["domain_classifier"].items()})
    print("policy", {k: {a: round(b, 3) for a, b in v.items()} for k, v in pol.items()})
    for dom, r in res["domains"].items():
        print(dom, f"test n={r['n_test']} malignant={r['n_malignant']}")
        for k in ("global_threshold", "per_domain_known", "per_domain_predicted_domain"):
            x = r[k]
            print(f"  {k:28s} sens {x['sensitivity']:.3f} [{x['sensitivity_ci95'][0]:.2f},{x['sensitivity_ci95'][1]:.2f}]  spec {x['specificity']:.3f}  ECE {x['ece']:.3f}")


if __name__ == "__main__":
    main()
