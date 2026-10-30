"""
Refit of the darker-skin scope detector and the phone-domain router with SCIN (classifier plan, routing check 2026-10-23:
the detector flagged only 34% of the FST V-VI atlas images and 21% of internal test images).

Variants (all L2 logistic regression on frozen DINOv2-L CLS; threshold from VALIDATION rows for 85% sensitivity to dark skin):
  base      ISIC + MIDAS tone labels (the adopted detector, refit here for a like-for-like comparison)
  +scin     base + SCIN train rows (dermatologist FST, case-grouped split splits/scin_v1.csv)
Router (close-up vs phone, train rows): base = v2 train vs MIDAS train; +scin = SCIN train rows added to the phone class.
Framing (report only): 3-way close-up / at an angle / at a distance head on SCIN shot_type.
Never reads the locked sets. DSD is a dev set (flagged share on its FST 5-6 rows is the target; v2_test share is the cost).

Adoption rule for +scin, fixed in the plan decision log on 2026-10-30 before this was run (all four must hold):
  SCIN-test sensitivity to dark skin >= 0.85; AUROC on the existing tone test rows (close-up, phone) >= base - 0.02;
  flagged share of DSD FST 5-6 above base; flagged share of split-v2 test <= 0.25.

    python eval/refit_detectors_scin.py        -> eval/output/sprint2/refit_scin.json (+ interpret tone_model_scin.npz if --write)
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
TONE = ROOT / "eval/output/interpretability/skin_tone"


def load_emb():
    emb = {}
    for f in ("dinov2.npz", "dinov2__midas.npz", "dinov2__dsd.npz", "dinov2__scin.npz"):
        z = np.load(EMB / f, allow_pickle=True)
        emb.update(dict(zip(z["isic_id"].astype(str), z["emb"].astype(np.float32))))
    return emb


def fit_tone(X, y, tr, va, C=0.003):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X))[:, 1]
    thr = float(np.quantile(p[va & (y == 1)], 0.15))
    return sc, m, thr


def main(opts):
    emb = load_emb()
    ita = pd.read_parquet(TONE / "ita.parquet")
    lab = ita[ita.dark.notna()][["id", "set", "domain", "dark"]]
    lab = lab.assign(id=lab.id.astype(str))
    scin = pd.read_csv(ROOT / "splits/scin_v1.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    scin = scin[scin.image_id.isin(emb.keys())]
    sl = scin[scin.dark.notna()]
    s = pd.DataFrame({"id": sl.image_id, "set": sl.split, "domain": "scin", "dark": sl.dark.astype(float)})
    dsd = pd.read_csv(ROOT / "splits/f17k_dsd_v1.csv", dtype={"isic_id": str})
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv", dtype={"image_id": str})
    mid = mid[mid.modality.str.startswith("phone")]
    probes = {"v2_test": list(v2[v2.split == "test"].isic_id), "midas_dev_test": list(mid[mid.split == "dev_test"].image_id),
              "dsd_fst56": list(dsd[dsd.dsd].isic_id), "dsd_fst4": list(dsd[~dsd.dsd].isic_id),
              "scin_test_dark": list(sl[(sl.split == "test") & (sl.dark == 1)].image_id),
              "scin_test_light": list(sl[(sl.split == "test") & (sl.dark == 0)].image_id)}
    res = {"tone": {}, "router": {}}
    for name, d in (("base", lab), ("+scin", pd.concat([lab, s], ignore_index=True))):
        X = np.stack([emb[i] for i in d.id])
        y = d.dark.astype(int).to_numpy()
        tr, va = (d.set == "train").to_numpy(), (d.set == "val").to_numpy()
        sc, m, thr = fit_tone(X, y, tr, va)
        flag = lambda ids: float((m.predict_proba(sc.transform(np.stack([emb[i] for i in ids])))[:, 1] >= thr).mean())
        r = {"threshold": thr, "flagged_share": {k: flag(v) for k, v in probes.items()}, "by_set": {}}
        p = m.predict_proba(sc.transform(X))[:, 1]
        for dom in ("close-up", "phone", "scin"):
            k = ((d.set == "test") & (d.domain == dom)).to_numpy()
            if k.any() and len(set(y[k])) == 2:
                r["by_set"][f"test/{dom}"] = {"n": int(k.sum()), "n_dark": int(y[k].sum()), "auroc": float(roc_auc_score(y[k], p[k])),
                                              "sens": float((p[k][y[k] == 1] >= thr).mean()), "spec": float((p[k][y[k] == 0] < thr).mean())}
        res["tone"][name] = r
        if name == "+scin" and opts.write:
            np.savez_compressed(TONE / "tone_model_scin.npz", mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0],
                                intercept=m.intercept_[0], threshold=thr)
    b, t = res["tone"]["base"], res["tone"]["+scin"]
    checks = {"scin_test_sensitivity>=0.85": t["by_set"]["test/scin"]["sens"] >= 0.85,
              "old_test_auroc_drop<=0.02": all(t["by_set"][k]["auroc"] >= v["auroc"] - 0.02
                                               for k, v in b["by_set"].items() if k != "test/scin"),
              "dsd_fst56_flagged_above_base": t["flagged_share"]["dsd_fst56"] > b["flagged_share"]["dsd_fst56"],
              "v2_test_flagged<=0.25": t["flagged_share"]["v2_test"] <= 0.25}
    res["tone"]["adoption"] = {"checks": checks, "adopt_scin": all(checks.values())}
    sh = scin[scin.shot_type.notna()]
    trf, tef = sh[sh.split == "train"], sh[sh.split == "test"]
    Xf, Xt = np.stack([emb[i] for i in trf.image_id]), np.stack([emb[i] for i in tef.image_id])
    scf = StandardScaler().fit(Xf)
    fr = LogisticRegression(C=0.01, max_iter=5000, class_weight="balanced").fit(scf.transform(Xf), trf.shot_type)
    pr = fr.predict(scf.transform(Xt))
    res["framing"] = {"test_n": len(tef), "accuracy": float((pr == tef.shot_type.to_numpy()).mean()),
                      "confusion": pd.crosstab(tef.shot_type.to_numpy(), pr).to_dict()}
    close = list(v2[v2.split == "train"].isic_id)
    phone = list(mid[mid.split == "train"].image_id)
    scin_tr = list(scin[scin.split == "train"].image_id)
    for name, ph in (("base", phone), ("+scin", phone + scin_tr)):
        ids = close + ph
        yv = np.r_[np.zeros(len(close)), np.ones(len(ph))]
        X = np.stack([emb[i] for i in ids])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(X), yv)
        res["router"][name] = {k: float((clf.predict_proba(sc.transform(np.stack([emb[i] for i in v])))[:, 1] >= 0.5).mean())
                               for k, v in probes.items()}
    out = ROOT / "eval/output/sprint2/refit_scin.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
