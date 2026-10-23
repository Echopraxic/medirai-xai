"""
Skin-tone scope rule: "darker skin: not validated" (decision 2026-10-22: the final model is near chance on Fitzpatrick
V-VI, DDI AUROC 0.58, so such images are flagged instead of explained as if the model were reliable).

Two estimators were tried, both validated against the tone labels we have (never DDI):
  ITA from pixels      NEGATIVE RESULT: AUROC 0.69-0.73 and a cutoff that flags 55% of all images (camera, lighting and
                       white balance differ between sources). Kept below for the record, not used.
  learned (adopted)    L2 logistic regression on frozen DINOv2-L features, trained on train rows (class-balanced),
                       threshold chosen on VALIDATION for 85% sensitivity to dark skin; test AUROC 0.91.
                       `python interpret/skin_tone.py fit` writes tone_model.npz + validation_learned.json.

ITA details (the negative result):

ITA = arctan((L* - 50) / b*) in degrees, from skin pixels (YCbCr skin mask) with the darkest 20% of them dropped
(lesion, shadow, hair), medians of L* and b*. Computed on a 256-px copy. Classical, transparent, no learned model.
Standard bands: > 55 very light, 41-55 light, 28-41 intermediate, 10-28 tan, -30-10 brown, < -30 dark.

Validation uses TRAIN rows only for choosing the cutoff, then reports on the selection data and the test split:
  dark label = Fitzpatrick V-VI (ISIC column, MIDAS) or MILK10k tone class <= 2 (0 = darkest)
  cutoff     ITA value maximising sensitivity + specificity (Youden) for "dark" on train
Never reads DDI, DDI-2, MCR-SL or the ISIC overview (locked): the rule is justified by the already-logged DDI result.

    python interpret/skin_tone.py            -> eval/output/interpretability/skin_tone/{ita.parquet,validation.json}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/output/interpretability/skin_tone"
ISIC_DIR = ROOT / "uncertaintyNet-main/datasets_768/ISIC_clinical"
MIDAS_DIR = ROOT / "data/midas/files_768"
SIZE = 256


def estimate_ita(img):
    """ITA in degrees for a PIL RGB image; NaN when too few skin pixels."""
    im = img.convert("RGB")
    im.thumbnail((SIZE, SIZE))
    ycc = np.asarray(im.convert("YCbCr")).astype(np.int32)
    cb, cr = ycc[..., 1], ycc[..., 2]
    skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    if skin.sum() < 500:
        return float("nan")
    lab = rgb2lab(np.asarray(im).astype(np.float64) / 255.0)[skin]
    keep = lab[:, 0] >= np.quantile(lab[:, 0], 0.2)
    lab = lab[keep]
    L, b = np.median(lab[:, 0]), np.median(lab[:, 2])
    return float(np.degrees(np.arctan2(L - 50.0, b)))


def labelled_images():
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    v2 = v2[v2.split.isin(["train", "val", "test"])]
    fz = pd.read_csv(ROOT / "splits/sources/isic_clinical_fitzpatrick.csv")
    milk = pd.read_csv(ROOT / "data/milk10k/supplements/training_input.csv")[["isic_id", "skin_tone_class"]].drop_duplicates("isic_id")
    d = v2.merge(fz, on="isic_id", how="left").merge(milk, on="isic_id", how="left")
    fst = d.fitzpatrick_skin_type.map({"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6})
    dark = np.where(fst.notna(), fst >= 5, np.where(d.skin_tone_class.notna(), d.skin_tone_class <= 2, np.nan))
    a = pd.DataFrame({"id": d.isic_id, "set": d.split, "domain": "close-up", "path": [str(ISIC_DIR / f"{i}.jpg") for i in d.isic_id],
                      "dark": dark, "source": d.source})
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    m = m[m.modality.str.startswith("phone") & m.split.isin(["train", "dev_val", "dev_test"])]
    mf = m.midas_fitzpatrick.astype(str).str.extract(r"^(vi|v|iv|iii|ii|i)\b")[0].map({"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6})
    b = pd.DataFrame({"id": m.image_id, "set": m.split.map({"train": "train", "dev_val": "val", "dev_test": "test"}),
                      "domain": "phone", "path": [str(MIDAS_DIR / f"{i}.jpg") for i in m.image_id],
                      "dark": (mf >= 5).astype(float).where(mf.notna()).to_numpy(), "source": "MIDAS"})
    return pd.concat([a, b], ignore_index=True)


def fit_tone_classifier():
    """Learned tone-scope model. Dark label: Fitzpatrick V-VI or MILK10k tone class <= 2."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    d = pd.read_parquet(OUT / "ita.parquet")
    emb = {}
    for f in ("dinov2.npz", "dinov2__midas.npz"):
        z = np.load(ROOT / "classifier/output/embeddings" / f, allow_pickle=True)
        emb.update(dict(zip(z["isic_id"], z["emb"].astype(np.float32))))
    lab = d[d.dark.notna()].reset_index(drop=True)
    X = np.stack([emb[i] for i in lab.id])
    y = lab.dark.astype(int).to_numpy()
    tr, va = (lab.set == "train").to_numpy(), (lab.set == "val").to_numpy()
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=0.003, max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
    p = m.predict_proba(sc.transform(X))[:, 1]
    thr = float(np.quantile(p[va & (y == 1)], 0.15))          # 85% sensitivity on validation dark images
    res = {"threshold": thr, "target_sensitivity_on_val": 0.85, "by_set": {}}
    for s in ("train", "val", "test"):
        k = (lab.set == s).to_numpy()
        flag = p[k] >= thr
        res["by_set"][s] = {"n": int(k.sum()), "n_dark": int(y[k].sum()), "auroc": float(roc_auc_score(y[k], p[k])),
                            "sensitivity_dark": float(flag[y[k] == 1].mean()), "specificity": float((~flag[y[k] == 0]).mean()),
                            "flagged_share": float(flag.mean())}
    k = lab.set.isin(["val", "test"]).to_numpy()
    res["by_source_val_test"] = {src: {"n": int(len(g)), "n_dark": int(g.dark.sum()), "flagged_share": float((p[g.index] >= thr).mean())}
                                 for src, g in lab[k].groupby("source")}
    np.savez_compressed(OUT / "tone_model.npz", mean=sc.mean_, scale=sc.scale_, coef=m.coef_[0], intercept=m.intercept_[0], threshold=thr)
    (OUT / "validation_learned.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


class ToneScope:
    """Runtime rule: DINOv2-L CLS embedding -> (flag, probability of 'darker skin')."""
    def __init__(self):
        z = np.load(OUT / "tone_model.npz")
        self.m = {k: z[k] for k in z.files}

    def __call__(self, dinov2_emb):
        x = (np.asarray(dinov2_emb, dtype=np.float64) - self.m["mean"]) / self.m["scale"]
        p = float(1 / (1 + np.exp(-(x @ self.m["coef"] + self.m["intercept"]))))
        return p >= float(self.m["threshold"]), p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = labelled_images()
    d["ita"] = [estimate_ita(Image.open(p)) for p in d.path]
    d.drop(columns="path").to_parquet(OUT / "ita.parquet", index=False)
    lab = d[d.dark.notna() & d.ita.notna()]
    tr = lab[lab.set == "train"]
    thr = np.sort(tr.ita.unique())
    best = max(((float((tr.ita[tr.dark == 1] <= t).mean() + (tr.ita[tr.dark == 0] > t).mean() - 1), float(t)) for t in thr))
    cut = best[1]
    res = {"cutoff_ita": cut, "youden_train": best[0], "nan_share": float(d.ita.isna().mean()), "by_set": {}}
    for s in ("train", "val", "test"):
        x = lab[lab.set == s]
        for dom in ("close-up", "phone", "all"):
            y = x if dom == "all" else x[x.domain == dom]
            if y.dark.nunique() < 2:
                continue
            flag = y.ita <= cut
            res["by_set"][f"{s}/{dom}"] = {
                "n": int(len(y)), "n_dark": int(y.dark.sum()), "auroc_dark_vs_ita": float(roc_auc_score(y.dark, -y.ita)),
                "sensitivity_dark": float(flag[y.dark == 1].mean()), "specificity": float((~flag[y.dark == 0]).mean()),
                "flagged_share": float(flag.mean())}
    allv = d[d.ita.notna() & d.set.isin(["val", "test"])]
    res["flagged_share_unlabelled_included"] = {s: float((allv.ita[allv.set == s] <= cut).mean()) for s in ("val", "test")}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    fit_tone_classifier() if sys.argv[1:] == ["fit"] else main()
