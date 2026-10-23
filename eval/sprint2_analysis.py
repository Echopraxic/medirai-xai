"""
Classifier sprint part 2 (prereg 2026-10-23): applies sections 2-6 to the per-run TTA scores (eval/tta_scores.py).

  arms       per-seed selection metrics (S, S3, DSD AUROC, val sensitivity/specificity, pooled melanoma recall) for
             S0 / S2 / S4, seed means and SDs, and the seed-ensemble paired bootstrap of the DSD AUROC difference
             against S0; adoption by the prereg rule
  candidate  E1 (best seed by S) vs E5 (5-seed mean) of the adopted recipe: temperature, lesion averaging, standard
             and melanoma-sensitive thresholds on internal val; the chosen one is reported once on v2 test / MIDAS dev_test
  deferral   rule D (out of scope OR max(entropy rank, seed-disagreement rank) > c, 30% on internal val) vs entropy only
  domains    close-up / phone / darker-skin routing from frozen DINOv2-L, per-domain temperature + conformal threshold

Image-level probabilities for S (as eval/selection_score.py); the product pipeline (calibration, lesion averaging) for
the candidate, deferral and domain sections. No locked set is read.

    python eval/sprint2_analysis.py --arm S0=<run>,<run>,... --arm S2=... --arm S4=... --out eval/output/sprint2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "interpret"))
from domain_policy import conformal_threshold  # noqa: E402
from final_model_report import ece, entropy, fit_temperature  # noqa: E402
from selection_score import threshold_for_sensitivity  # noqa: E402
from stacking import SCOPE, mahalanobis  # noqa: E402

SETS = ("v2_val", "v2_test", "midas_dev_val", "midas_dev_test", "dsd")
EMB = ROOT / "classifier/output/embeddings"
N_BOOT = 2000


def sig(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------ data
def meta():
    """Per set: id-indexed frame with label, melanoma, group (bootstrap unit) and lesion (averaging unit)."""
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone")]
    dsd = pd.read_csv(ROOT / "splits/f17k_dsd_v1.csv")
    out = {}
    for s, split in (("v2_val", "val"), ("v2_test", "test")):
        x = v2[v2.split == split]
        out[s] = pd.DataFrame({"label": x.label.astype(int).values, "group": x.group_id.values,
                               "melanoma": x.diagnosis_3.fillna("").str.contains("Melanoma").values,
                               "lesion": x.lesion_id.fillna(x.isic_id).values}, index=x.isic_id.values)
    for s, split in (("midas_dev_val", "dev_val"), ("midas_dev_test", "dev_test")):
        x = mid[mid.split == split]
        out[s] = pd.DataFrame({"label": x.label.astype(int).values, "group": x.patient_id.values,
                               "melanoma": x.midas_path.str.contains("melanoma").values,
                               "lesion": x.lesion_id.values}, index=x.image_id.values)
    out["dsd"] = pd.DataFrame({"label": dsd.label.astype(int).values, "group": dsd.group_id.values,
                               "melanoma": dsd.diagnosis_3.str.contains("Melanoma").values, "lesion": dsd.group_id.values,
                               "dsd": dsd.dsd.astype(bool).values}, index=dsd.isic_id.values)
    return out


def load_run(run):
    z = np.load(UNC / "output" / run / "tta_scores.npz", allow_pickle=True)
    return {s: pd.DataFrame({"margin": z[f"{s}_margin"], "pmel": z[f"{s}_pmel"] if f"{s}_pmel" in z.files else np.nan},
                            index=z[f"{s}_ids"]) for s in SETS}


def embeddings():
    emb = {}
    for f in ("dinov2.npz", "dinov2__midas.npz", "dinov2__dsd.npz"):
        z = np.load(EMB / f, allow_pickle=True)
        emb.update({k: v for k, v in zip(z["isic_id"], z["emb"].astype(np.float64))})
    return emb


# ------------------------------------------------------------------------------------------------ metrics
def auroc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def selection(p, M):
    """Image-level selection metrics of one probability set {set: array aligned with M[set]}."""
    y = {s: M[s].label.values for s in M}
    dsd = M["dsd"].dsd.values
    r = {"auroc_v2_val": auroc(y["v2_val"], p["v2_val"]), "auroc_midas_dev_val": auroc(y["midas_dev_val"], p["midas_dev_val"]),
         "auroc_dsd": auroc(y["dsd"][dsd], p["dsd"][dsd]), "auroc_dsd_iv": auroc(y["dsd"], p["dsd"])}
    r["S"] = (r["auroc_v2_val"] + r["auroc_midas_dev_val"]) / 2
    r["S3"] = (r["auroc_v2_val"] + r["auroc_midas_dev_val"] + r["auroc_dsd"]) / 3
    t = threshold_for_sensitivity(y["v2_val"], p["v2_val"])
    pv = p["v2_val"] >= t
    r["val_sensitivity"] = float(pv[y["v2_val"] == 1].mean())
    r["val_specificity"] = float((~pv[y["v2_val"] == 0]).mean())
    mel = np.concatenate([pv[M["v2_val"].melanoma.values], (p["midas_dev_val"] >= t)[M["midas_dev_val"].melanoma.values]])
    r["pooled_melanoma_recall"], r["n_pooled_melanoma"] = float(mel.mean()), int(len(mel))
    dm = M["dsd"].melanoma.values & dsd
    r["dsd_melanoma_recall"] = float((p["dsd"][dm] >= t).mean()) if dm.any() else None
    return r


def paired_auroc_diff(y, a, b, groups, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in g}
    diffs = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if len(set(y[take])) == 2:
            diffs.append(roc_auc_score(y[take], b[take]) - roc_auc_score(y[take], a[take]))
    return {"diff": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def raw_probs(scores, M):
    return {s: sig(scores[s].margin.loc[M[s].index].values) for s in SETS}


# ------------------------------------------------------------------------------------------------ sections
def arms(runs, M):
    res, ens = {}, {}
    for arm, rs in runs.items():
        per = [selection(raw_probs(load_run(r), M), M) for r in rs]
        df = pd.DataFrame(per, index=rs)
        res[arm] = {"seeds": df.to_dict(orient="index"), "mean": df.mean().to_dict(), "sd": df.std().to_dict()}
        ens[arm] = {s: np.mean([raw_probs(load_run(r), M)[s] for r in rs], axis=0) for s in SETS}
        res[arm]["seed_ensemble"] = selection(ens[arm], M)
    d = M["dsd"].dsd.values
    y, g = M["dsd"].label.values[d], M["dsd"].group.values[d]
    ref = res["S0"]["mean"]
    adopted = []
    for arm in runs:
        if arm == "S0":
            continue
        m = res[arm]["mean"]
        boot = paired_auroc_diff(y, ens["S0"]["dsd"][d], ens[arm]["dsd"][d], g)
        checks = {"dsd_gain_ge_0.03": m["auroc_dsd"] - ref["auroc_dsd"] >= 0.03, "dsd_ci_excludes_0": boot["ci95"][0] > 0,
                  "S_not_worse_0.005": m["S"] >= ref["S"] - 0.005,
                  **{f"{k}_not_worse_0.02": m[k] >= ref[k] - 0.02
                     for k in ("val_sensitivity", "val_specificity", "pooled_melanoma_recall")}}
        res[arm]["vs_S0"] = {"dsd_auroc_gain_seed_mean": m["auroc_dsd"] - ref["auroc_dsd"], "dsd_seed_ensemble_bootstrap": boot,
                             "S_gain": m["S"] - ref["S"], "checks": checks, "adopt": all(checks.values())}
        if all(checks.values()):
            adopted.append((m["auroc_dsd"] - ref["auroc_dsd"], arm))
    res["adopted"] = max(adopted)[1] if adopted else "S0"
    return res


def lesion_mean(p, M_s):
    return pd.Series(p, index=M_s.index).groupby(M_s.lesion.values).transform("mean").values


def candidate(members, M):
    """Calibrated, lesion-averaged probabilities and thresholds of the mean of the members' TTA probabilities."""
    sc = [load_run(r) for r in members]
    p_img = {s: np.mean([sig(x[s].margin.loc[M[s].index].values) for x in sc], axis=0) for s in SETS}
    pmel = {s: np.mean([x[s].pmel.loc[M[s].index].values for x in sc], axis=0) for s in SETS}
    T = fit_temperature(logit(p_img["v2_val"]), M["v2_val"].label.values)
    p_cal = {s: sig(logit(p_img[s]) / T) for s in SETS}
    p_dec = {s: lesion_mean(p_cal[s], M[s]) for s in SETS}
    pmel = {s: lesion_mean(pmel[s], M[s]) for s in SETS}
    y, mel = M["v2_val"].label.values, M["v2_val"].melanoma.values
    t = threshold_for_sensitivity(y, p_dec["v2_val"])
    tm, best = None, None
    if not np.isnan(pmel["v2_val"]).any():
        for cand in np.quantile(pmel["v2_val"], np.linspace(0.3, 0.995, 140)):
            pred = (p_dec["v2_val"] >= t) | (pmel["v2_val"] >= cand)
            if pred[mel].mean() >= 0.65 and (best is None or (~pred[y == 0]).mean() > best):
                tm, best = float(cand), float((~pred[y == 0]).mean())
    return {"members": members, "T": T, "t": t, "tm": tm, "p_img": p_img, "p_cal": p_cal, "p": p_dec, "pmel": pmel}


def operating(c, M, s):
    y, mel, p = M[s].label.values, M[s].melanoma.values, c["p"][s]
    if s == "dsd":
        keep = M[s].dsd.values
        y, mel, p = y[keep], mel[keep], p[keep]
    out = {"n": int(len(y)), "auroc": auroc(y, p), "ece": ece(p, y)}
    modes = {"standard": p >= c["t"]}
    if c["tm"] is not None:
        pm = c["pmel"][s][M[s].dsd.values] if s == "dsd" else c["pmel"][s]
        modes["melanoma_sensitive"] = (p >= c["t"]) | (pm >= c["tm"])
    for k, pred in modes.items():
        out[k] = {"sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
                  "melanoma_recall": float(pred[mel].mean()) if mel.any() else None}
    return out


def choose_candidate(recipe_runs, M):
    sel = {}
    for r in recipe_runs:
        sel[r] = selection(raw_probs(load_run(r), M), M)["S"]
    e1 = candidate([max(sel, key=sel.get)], M)
    e5 = candidate(recipe_runs, M)
    s1, s5 = selection(e1["p_img"], M), selection(e5["p_img"], M)
    v1, v5 = operating(e1, M, "v2_val"), operating(e5, M, "v2_val")
    checks = {"sensitivity": v5["standard"]["sensitivity"] >= v1["standard"]["sensitivity"] - 0.02,
              "pooled_melanoma_recall": s5["pooled_melanoma_recall"] >= s1["pooled_melanoma_recall"] - 0.02}
    better = (s5["S"] >= s1["S"] + 0.005) or (v5["standard"]["specificity"] >= v1["standard"]["specificity"] + 0.01)
    if v5["standard"]["specificity"] < v1["standard"]["specificity"] - 0.02:
        better = False
    chosen = e5 if (better and all(checks.values())) else e1
    res = {"E1": {"run": e1["members"][0], "selection": s1, "v2_val": v1, "T": e1["T"], "t": e1["t"], "tm": e1["tm"]},
           "E5": {"selection": s5, "v2_val": v5, "T": e5["T"], "t": e5["t"], "tm": e5["tm"]},
           "E5_checks": checks, "E5_better": bool(better), "chosen": "E5" if chosen is e5 else "E1"}
    res["chosen_report"] = {s: operating(chosen, M, s) for s in ("v2_val", "midas_dev_val", "dsd", "v2_test", "midas_dev_test")}
    return chosen, res


def rank_against(ref, x):
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / len(ref)


def deferral(chosen, recipe_runs, M, emb):
    seeds = [candidate([r], M) for r in recipe_runs]
    scope = np.load(SCOPE)
    out = {}
    sig_v = {"ent": entropy(chosen["p"]["v2_val"]), "dis": np.std([c["p"]["v2_val"] for c in seeds], axis=0)}
    per = {}
    for s in SETS:
        keep = M[s].dsd.values if s == "dsd" else np.ones(len(M[s]), bool)
        e = entropy(chosen["p"][s])
        dis = np.std([c["p"][s] for c in seeds], axis=0)
        u = np.maximum(rank_against(sig_v["ent"], e), rank_against(sig_v["dis"], dis))
        dist = mahalanobis(np.stack([emb[i] for i in M[s].index]), scope)
        err = ((chosen["p"][s] >= chosen["t"]).astype(int) != M[s].label.values)
        per[s] = {k: v[keep] for k, v in dict(e=e, u=u, out=dist > float(scope["cut"]), err=err).items()}
    v = per["v2_val"]
    cs = np.quantile(v["u"], np.linspace(0, 1, 2001))
    share = np.array([(v["out"] | (v["u"] > c)).mean() for c in cs])
    c = float(cs[np.argmin(np.abs(share - 0.30))])
    for s, x in per.items():
        d = x["out"] | (x["u"] > c)
        e30 = x["e"] > np.quantile(x["e"], 0.7)
        out[s] = {"n": int(len(d)), "error_rate": float(x["err"].mean()), "deferred_share": float(d.mean()),
                  "out_of_scope_share": float(x["out"].mean()),
                  "accuracy_retained_rule_D": float(1 - x["err"][~d].mean()) if (~d).any() else None,
                  "accuracy_retained_entropy_30pct": float(1 - x["err"][~e30].mean()),
                  "auroc_errors_u": auroc(x["err"], x["u"]), "auroc_errors_entropy": auroc(x["err"], x["e"])}
    gain = np.mean([out[s]["auroc_errors_u"] - out[s]["auroc_errors_entropy"] for s in ("midas_dev_val", "dsd")])
    v = per["v2_val"]
    u30 = v["u"] > np.quantile(v["u"], 0.7)
    acc_u30 = float(1 - v["err"][~u30].mean())
    out["cut_c"] = c
    out["adoption"] = {"mean_auroc_gain_midas_dsd": float(gain), "val_accuracy_u_30pct": acc_u30,
                       "val_accuracy_entropy_30pct": out["v2_val"]["accuracy_retained_entropy_30pct"],
                       "adopt": bool(gain >= 0.03 and acc_u30 >= out["v2_val"]["accuracy_retained_entropy_30pct"] - 0.005)}
    return out


def router(emb):
    """Domain per image id: 'dark' (learned tone rule) > 'phone' (close-up vs phone classifier) > 'close-up'."""
    from skin_tone import ToneScope
    tone = ToneScope()
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone")]
    ids = list(v2[v2.split == "train"].isic_id) + list(mid[mid.split == "train"].image_id)
    yv = np.r_[np.zeros((v2.split == "train").sum()), np.ones((mid.split == "train").sum())]
    X = np.stack([emb[i] for i in ids])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(X), yv)

    def route(id_list):
        E = np.stack([emb[i] for i in id_list])
        phone = clf.predict_proba(sc.transform(E))[:, 1] >= 0.5
        dark = np.array([tone(e)[0] for e in E])
        return np.where(dark, "dark", np.where(phone, "phone", "close-up"))
    return route


def domains(chosen, M, emb):
    route = router(emb)
    keep = M["dsd"].dsd.values
    fit_sets = {"close-up": ("v2_val", None), "phone": ("midas_dev_val", None), "dark": ("dsd", keep)}
    held = {"close-up": "v2_test", "phone": "midas_dev_test"}
    pol, res = {}, {"fits": {}, "held_out": {}, "routed": {}}
    for dom, (s, k) in fit_sets.items():
        p, y = chosen["p"][s], M[s].label.values
        if k is not None:
            p, y = p[k], y[k]
        T = fit_temperature(logit(p), y)
        t = conformal_threshold(sig(logit(p) / T), y)
        pol[dom] = (T, t)
        res["fits"][dom] = {"T": T, "t": t, "n": int(len(y))}
    for dom, s in held.items():
        p, y = chosen["p"][s], M[s].label.values
        T, t = pol[dom]
        q = sig(logit(p) / T)
        res["held_out"][dom] = {"set": s, "ece_global": ece(p, y), "ece_domain": ece(q, y),
                                "sens_domain": float((q >= t)[y == 1].mean()), "spec_domain": float((q < t)[y == 0].mean())}
    # darker skin: 2-fold cross-fit by duplicate group on DSD (FST 5-6)
    p, y, g = chosen["p"]["dsd"][keep], M["dsd"].label.values[keep], M["dsd"].group.values[keep]
    fold = (pd.Series(g).astype("category").cat.codes.values % 2)
    q = np.empty_like(p)
    pred = np.empty(len(p), bool)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        T = fit_temperature(logit(p[tr]), y[tr])
        t = conformal_threshold(sig(logit(p[tr]) / T), y[tr])
        q[te] = sig(logit(p[te]) / T)
        pred[te] = q[te] >= t
    res["held_out"]["dark"] = {"set": "dsd (2-fold cross-fit)", "ece_global": ece(p, y), "ece_domain": ece(q, y),
                               "sens_domain": float(pred[y == 1].mean()), "spec_domain": float((~pred[y == 0]).mean())}
    for s in ("v2_test", "midas_dev_test", "dsd"):
        ids = M[s].index
        r = route(list(ids))
        p, y = chosen["p"][s], M[s].label.values
        T = np.array([pol[d][0] for d in r])
        t = np.array([pol[d][1] for d in r])
        q = sig(logit(p) / T)
        sel = M[s].dsd.values if s == "dsd" else np.ones(len(y), bool)
        res["routed"][s] = {"route_share": pd.Series(r[sel]).value_counts(normalize=True).to_dict(),
                            "ece_global": ece(p[sel], y[sel]), "ece_routed": ece(q[sel], y[sel]),
                            "sens_global_t": float((p[sel] >= chosen["t"])[y[sel] == 1].mean()),
                            "spec_global_t": float((p[sel] < chosen["t"])[y[sel] == 0].mean()),
                            "sens_routed": float((q[sel] >= t[sel])[y[sel] == 1].mean()),
                            "spec_routed": float((q[sel] < t[sel])[y[sel] == 0].mean())}
    res["note"] = "routed DSD numbers use the full-DSD darker-skin fit (in-sample for DSD); the cross-fit row is the honest one"
    return res


def main(opts):
    runs = dict(a.split("=", 1) for a in opts.arm)
    runs = {k: v.split(",") for k, v in runs.items()}
    M = meta()
    emb = embeddings()
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"arms": arms(runs, M)}
    recipe = runs[res["arms"]["adopted"]]
    chosen, res["candidate"] = choose_candidate(recipe, M)
    res["deferral"] = deferral(chosen, recipe, M, emb)
    res["domains"] = domains(chosen, M, emb)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(out / "chosen_scores.npz", **{f"{s}_p": chosen["p"][s] for s in SETS},
                        **{f"{s}_pmel": chosen["pmel"][s] for s in SETS}, T=chosen["T"], t=chosen["t"],
                        tm=chosen["tm"] if chosen["tm"] is not None else np.nan)
    print(json.dumps({"adopted_arm": res["arms"]["adopted"], "candidate": res["candidate"]["chosen"],
                      "deferral_adopt": res["deferral"]["adoption"], **{a: res["arms"][a]["mean"] for a in runs}},
                     indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=run1,run2,... (runs under uncertaintyNet-main/output)")
    ap.add_argument("--out", default="eval/output/sprint2")
    main(ap.parse_args())
