"""
Step 2 of the PASSION rule (plan decision 2026-11-04, fixed before any PASSION score was computed): runs only if the
adopted detector's PASSION-test sensitivity to FST V-VI (eval/output/passion/tone_check.json) is below 0.85.

  adopted         the +SCIN per-domain detector, refit here exactly as eval/refit_detectors_scin.py does (checked
                  against tone_model_scin_domain.npz)
  +passion/domain adopted + PASSION train rows (dark = FST V-VI); router unchanged (+SCIN router); one threshold per
                  route at 85% sensitivity on that route's validation dark rows (PASSION val included)

Adopt only if all hold: PASSION-test V-VI sensitivity >= 0.85; SCIN-test dark sensitivity >= 0.85; tone-test AUROC per
domain (close-up, phone, SCIN) >= adopted - 0.02; DSD v2 FST V-VI flagged share >= adopted - 0.02; split-v2 test
flagged <= 0.25. Reported only: MIDAS dev_test and SCIN light flagged shares, PASSION III-IV specificity. One attempt.
Never reads DDI or any locked set; the classifier is not touched. Everything runs on the laptop (PASSION stays local).

    python eval/refit_detector_passion.py [--write]   -> eval/output/passion/refit.json (+ tone_model_passion_domain.npz)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval.refit_detectors_scin import EMB, TONE, by_set, fit_router, fit_tone, load_emb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/output/passion"


def tone_rows(emb):
    """Existing tone labels (ISIC, MIDAS) + SCIN, as in eval/refit_detectors_scin.py; + PASSION rows separately."""
    ita = pd.read_parquet(TONE / "ita.parquet")
    lab = ita[ita.dark.notna()][["id", "set", "domain", "dark"]]
    lab = lab.assign(id=lab.id.astype(str))
    scin = pd.read_csv(ROOT / "splits/scin_v1.csv", dtype={"image_id": str}).drop_duplicates("image_id")
    scin = scin[scin.image_id.isin(emb.keys())]
    sl = scin[scin.dark.notna()]
    s = pd.DataFrame({"id": sl.image_id, "set": sl.split, "domain": "scin", "dark": sl.dark.astype(float)})
    pa = pd.read_csv(ROOT / "splits/passion_v1.csv", dtype={"image_id": str, "subject_id": str})
    p = pd.DataFrame({"id": pa.image_id, "set": pa.split, "domain": "passion", "dark": pa.dark.astype(float),
                      "fitzpatrick": pa.fitzpatrick})
    return pd.concat([lab, s], ignore_index=True), p, scin, sl


def main(opts):
    check = json.loads((OUT / "tone_check.json").read_text())
    sens = check["test"]["sensitivity_dark"]
    if sens >= 0.85:
        print(f"PASSION-test sensitivity {sens:.3f} >= 0.85: step 2 does not run (rule 2026-11-04)")
        return
    emb = load_emb()
    z = np.load(EMB / "dinov2__passion.npz", allow_pickle=True)
    emb.update(dict(zip(z["isic_id"].astype(str), z["emb"].astype(np.float32))))
    base, pas, scin, sl = tone_rows(emb)
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv", dtype={"image_id": str})
    mid = mid[mid.modality.str.startswith("phone")]
    dsd2 = pd.read_csv(ROOT / "splits/f17k_dsd_v2.csv", dtype={"isic_id": str})
    probes = {"v2_test": list(v2[v2.split == "test"].isic_id), "midas_dev_test": list(mid[mid.split == "dev_test"].image_id),
              "dsd_v2_fst56": list(dsd2.isic_id),
              "scin_test_light": list(sl[(sl.split == "test") & (sl.dark == 0)].image_id),
              "passion_test_fst56": list(pas[(pas.set == "test") & (pas.dark == 1)].id),
              "passion_test_fst34": list(pas[(pas.set == "test") & (pas.dark == 0)].id)}
    close = list(v2[v2.split == "train"].isic_id)
    phone = list(mid[mid.split == "train"].image_id)
    router = fit_router(emb, close, phone + list(scin[scin.split == "train"].image_id))
    res = {"precondition": {"passion_test_sensitivity_adopted": sens}, "tone": {}}
    fits = {}
    for name, d in (("adopted", base), ("+passion/domain", pd.concat([base, pas.drop(columns="fitzpatrick")], ignore_index=True))):
        X = np.stack([emb[i] for i in d.id])
        y = d.dark.astype(int).to_numpy()
        tr, va = (d.set == "train").to_numpy(), (d.set == "val").to_numpy()
        sc, m, _ = fit_tone(X, y, tr, va)
        p = m.predict_proba(sc.transform(X))[:, 1]
        ph = router(list(d.id))
        thr = {r: float(np.quantile(p[va & (y == 1) & (ph == r)], 0.15)) for r in (False, True)}
        score = lambda ids, sc=sc, m=m: m.predict_proba(sc.transform(np.stack([emb[i] for i in ids])))[:, 1]
        row_thr = lambda ids, thr=thr: np.where(router(ids), thr[True], thr[False])
        flag = lambda ids, score=score, row_thr=row_thr: score(ids) >= row_thr(ids)
        pt = pas[pas.set == "test"]
        res["tone"][name] = {
            "threshold": {"close-up": thr[False], "phone": thr[True]},
            "flagged_share": {k: float(flag(v).mean()) for k, v in probes.items()},
            "by_set": by_set(d, y, p, row_thr(list(d.id))),
            "passion_test": {"auroc": float(roc_auc_score(pt.dark, score(list(pt.id)))),
                             "flagged_by_fst": {int(f): float(flag(list(g.id)).mean()) for f, g in pt.groupby("fitzpatrick")}}}
        fits[name] = (sc, m, thr)
    saved = np.load(TONE / "tone_model_scin_domain.npz")
    a = res["tone"]["adopted"]["threshold"]
    res["adopted_matches_saved_model"] = bool(abs(a["close-up"] - float(saved["threshold_closeup"])) < 1e-6
                                              and abs(a["phone"] - float(saved["threshold_phone"])) < 1e-6)
    A, V = res["tone"]["adopted"], res["tone"]["+passion/domain"]
    checks = {"passion_test_sensitivity>=0.85": V["flagged_share"]["passion_test_fst56"] >= 0.85,
              "scin_test_sensitivity>=0.85": V["by_set"]["test/scin"]["sens"] >= 0.85,
              "tone_test_auroc_drop<=0.02": all(V["by_set"][k]["auroc"] >= A["by_set"][k]["auroc"] - 0.02
                                                for k in ("test/close-up", "test/phone", "test/scin")),
              "dsd_v2_fst56_flagged>=adopted-0.02": V["flagged_share"]["dsd_v2_fst56"] >= A["flagged_share"]["dsd_v2_fst56"] - 0.02,
              "v2_test_flagged<=0.25": V["flagged_share"]["v2_test"] <= 0.25}
    res["adoption +passion/domain"] = {"checks": checks, "adopt": all(checks.values())}
    if opts.write and all(checks.values()):
        sc, m, thr = fits["+passion/domain"]
        np.savez_compressed(TONE / "tone_model_passion_domain.npz", mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0],
                            intercept=m.intercept_[0], threshold_closeup=thr[False], threshold_phone=thr[True],
                            router_mean=router.sc.mean_, router_scale=router.sc.scale_,
                            router_coef=router.clf.coef_[0], router_intercept=router.clf.intercept_[0])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "refit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="save tone_model_passion_domain.npz if adopted")
    main(ap.parse_args())
