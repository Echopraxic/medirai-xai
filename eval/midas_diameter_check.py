"""
First calibrated check of the diameter proxy (plan WS3, "D"): MRA-MIDAS phone photos are taken at fixed distances
(6 in, 1 ft) and come with the lesion's measured length and width in mm. Our only size measure is the max Feret
diameter of the UNet mask relative to the image diagonal (features/extract.frame_features, QA only, not a model input).

Per distance: Spearman of frame-relative Feret vs measured length (95% CI over patients), and a log-log fit
log(length mm) ~ log(frame Feret) with 5-fold patient-grouped cross-validation (median absolute % error of the
predicted length). Selection data rules do not apply (nothing is chosen); MIDAS dev_test rows are included because
no model is scored.

    python eval/midas_diameter_check.py   ->  eval/output/midas_diameter/check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
from extract import frame_features  # noqa: E402

MASKS = ROOT / "data/midas/masks_unet_clinical"
N_BOOT = 2000


def cv_error(x, y, groups):
    """Median absolute % error of exp(a + b log x) for length y, 5-fold by patient."""
    lx, ly, pred = np.log(x), np.log(y), np.empty(len(y))
    for tr, te in GroupKFold(5).split(lx, ly, groups):
        b, a = np.polyfit(lx[tr], ly[tr], 1)
        pred[te] = a + b * lx[te]
    return float(np.median(np.abs(np.exp(pred) - y) / y))


def main():
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    m = m[m.modality.isin(["phone_6in", "phone_1ft"]) & (m.length__mm_ > 0)]
    rows = []
    for r in m.itertuples():
        p = MASKS / f"{r.image_id}.png"
        if not p.exists():
            continue
        mask = np.array(Image.open(p)) > 0
        if mask.sum() < 20:
            continue
        rows.append({"modality": r.modality, "patient": r.patient_id, "length_mm": r.length__mm_,
                     "width_mm": r.width__mm_, **frame_features(mask)})
    d = pd.DataFrame(rows)
    out = {"n_images": len(d)}
    rng = np.random.default_rng(0)
    for mod, g in d.groupby("modality"):
        pats = g.patient.unique()
        idx = {p: np.flatnonzero(g.patient.values == p) for p in pats}
        boots = []
        for _ in range(N_BOOT):
            take = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
            boots.append(spearmanr(g.frame_feret_rel.values[take], g.length_mm.values[take]).statistic)
        out[mod] = {"n": len(g), "n_patients": len(pats), "length_mm_median": float(g.length_mm.median()),
                    "spearman_feret_vs_length": float(spearmanr(g.frame_feret_rel, g.length_mm).statistic),
                    "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                    "cv_median_abs_pct_error_length": cv_error(g.frame_feret_rel.values, g.length_mm.values, g.patient.values),
                    "baseline_median_abs_pct_error": float(np.median(np.abs(g.length_mm.median() - g.length_mm) / g.length_mm))}
    path = ROOT / "eval/output/midas_diameter/check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
