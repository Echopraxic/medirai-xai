"""
Concept validation against ENHANCE (plan T3.3, criterion C2): do our A/B/C features agree with human ratings?

ENHANCE (Raumanns et al., MELBA 2021; github.com/raumannsr/ENHANCE) rates ISIC 2017 dermoscopic images for
asymmetry (0-2), border irregularity (0-8) and number of colors (1-6):
  - crowd: 3 MTurk raters per image, 1,250 images (primary labels: one consistent scale)
  - student: undergraduate groups, 2017-2020 (secondary; only columns whose header is the English concept
    name are used, because some Dutch-headed columns are reverse-coded, e.g. "Egaal" = uniform)
  - automated: ENHANCE's own image-processing scores (a reference algorithm)
Labels are non-expert, so the rater-vs-rater agreement is reported as the ceiling a feature can reach.

Features are extracted with features/extract.py (same version as the feature table) on the ISIC 2017
ground-truth masks, so this measures concept validity separately from our segmentation quality.
Domain caveat: dermoscopy, while our target is clinical close-ups.

    python concepts/validate_enhance.py      # data/concepts/{ENHANCE,isic2017}; writes concepts/output/enhance_*
"""
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))
import extract as fx  # noqa: E402

DATA = ROOT / "data" / "concepts"
ENH = DATA / "ENHANCE" / "0_data"
IMAGES = DATA / "isic2017" / "ISIC-2017_Training_Data"
MASKS = DATA / "isic2017" / "ISIC-2017_Training_Part1_GroundTruth"
OUT = ROOT / "concepts" / "output"
CONCEPTS = {"asymmetry": "A", "border": "B", "color": "C"}
C2_TARGET = 0.5


def crowd_labels():
    frames = []
    for concept in CONCEPTS:
        df = pd.read_csv(ENH / "crowd" / f"{concept}.csv").set_index("ID")
        frames.append(df.add_prefix(f"{concept}_"))
    return pd.concat(frames, axis=1)


def student_labels():
    """Mean within-group percentile rank per concept (groups use different scales)."""
    per_concept = {c: [] for c in CONCEPTS}
    for path in sorted((ENH / "student").glob("*/group*.csv")):
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig").dropna(axis=1, how="all")
        df = df.set_index(df.columns[0])
        for concept in CONCEPTS:
            cols = [c for c in df.columns if c.lower().startswith(concept) and "categor" not in c.lower()
                    and "fade" not in c.lower()]
            if cols:
                vals = df[cols].apply(pd.to_numeric, errors="coerce")
                per_concept[concept].append(vals.rank(pct=True).mean(axis=1).rename(path.parent.name + path.stem))
    return pd.DataFrame({c: pd.concat(v, axis=1).mean(axis=1) for c, v in per_concept.items() if v})


def automated_labels():
    a = pd.read_csv(ENH / "automated" / "computer_asymmetry_0.9.csv").set_index("ID")["i"].rename("asymmetry")
    b = pd.read_csv(ENH / "automated" / "computer_border_.csv")
    b["ID"] = b.ID.str.replace("_segmentation", "")
    b = b.set_index("ID")["i"].rename("border")
    c = pd.read_csv(ENH / "automated" / "computer_color.csv").set_index("ID")["i"].rename("color")
    return pd.concat([a, b, c], axis=1)


def one(isic_id):
    try:
        img = np.array(Image.open(IMAGES / f"{isic_id}.jpg").convert("RGB"))
        mask = np.array(Image.open(MASKS / f"{isic_id}_segmentation.png")) > 127
        return {"ID": isic_id, **fx.extract(img, mask)}
    except Exception as err:
        return {"ID": isic_id, "error": f"{type(err).__name__}: {err}"}


def features(ids, workers=6):
    cache = OUT / f"isic2017_features_{fx.FEATURE_VERSION}.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("ID")
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(one, ids, chunksize=8))
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df.set_index("ID")


def rho(a, b):
    ok = a.notna() & b.notna()
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 10 else float("nan")


def rater_vs_features(crowd, concept, X):
    """Equal-footing comparison: for each held-out rater, the rater and the (cross-validated) feature model are
    both scored against the mean of the OTHER raters. Feature >= rater means the features agree with the
    consensus at least as well as a human rater does (relative C2)."""
    cols = [c for c in crowd.columns if c.startswith(concept)]
    pairs = [rho(crowd[a], crowd[b]) for a, b in itertools.combinations(cols, 2)]
    rater, feature = [], []
    for c in cols:
        others = crowd[[o for o in cols if o != c]].mean(axis=1).reindex(X.index)
        rater.append(rho(crowd[c].reindex(X.index), others))
        feature.append(group_model_rho(X, others)[0])
    return float(np.mean(pairs)), float(np.mean(rater)), float(np.mean(feature))


def group_model_rho(X, y, seed=0):
    """Cross-validated linear combination of one concept group's features (5-fold), Spearman with the label."""
    ok = y.notna() & X.notna().all(axis=1)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    pred = cross_val_predict(model, X[ok], y[ok], cv=KFold(5, shuffle=True, random_state=seed))
    return float(spearmanr(pred, y[ok]).statistic), int(ok.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crowd, student, auto = crowd_labels(), student_labels(), automated_labels()
    ids = sorted(i for i in set(crowd.index) | set(student.index) if isinstance(i, str) and i.startswith("ISIC_"))
    ids = [i for i in ids if (IMAGES / f"{i}.jpg").exists() and (MASKS / f"{i}_segmentation.png").exists()]
    feats = features(ids)
    fcols = [c for c in feats.columns if c in fx.FEATURE_DICTIONARY]
    report = {"feature_version": fx.FEATURE_VERSION, "n_images_with_features": int(feats[fcols].notna().all(axis=1).sum()),
              "c2_target_spearman": C2_TARGET, "concepts": {}}
    rows = []
    for concept, group in CONCEPTS.items():
        y_crowd = crowd[[c for c in crowd.columns if c.startswith(concept)]].mean(axis=1).reindex(feats.index)
        y_student = student[concept].reindex(feats.index) if concept in student else None
        group_cols = [c for c in fcols if c[0] == group]
        pair, loo, feat_loo = rater_vs_features(crowd, concept, feats[group_cols])
        single = {c: rho(feats[c], y_crowd) for c in group_cols}
        best = max(single, key=lambda c: abs(single[c]) if not np.isnan(single[c]) else -1)
        g_rho, n = group_model_rho(feats[group_cols], y_crowd)
        all_rho, _ = group_model_rho(feats[fcols], y_crowd)
        entry = {
            "n_crowd": n,
            "crowd_rater_pairwise_rho": pair, "crowd_rater_vs_others_rho": loo,
            "crowd_vs_student_rho": rho(y_crowd, y_student) if y_student is not None else None,
            "enhance_automated_vs_crowd_rho": rho(auto[concept].reindex(feats.index), y_crowd),
            "best_single_feature": best, "best_single_feature_rho": single[best],
            "group_model_cv_rho": g_rho, "all_features_model_cv_rho": all_rho,
            "group_model_vs_student_rho": (group_model_rho(feats[group_cols], y_student)[0]
                                           if y_student is not None else None),
            "features_vs_other_raters_rho": feat_loo,
            "meets_C2_absolute": bool(g_rho >= C2_TARGET),
            "meets_C2_relative": bool(feat_loo >= loo),
            "labels_usable": bool(loo >= 0.2),
            "single_feature_rho": single,
        }
        report["concepts"][concept] = entry
        rows.append({"concept": concept, "n": n, "rater_vs_other_raters": loo, "features_vs_other_raters": feat_loo,
                     "labels_usable (rater>=0.2)": entry["labels_usable"],
                     "features >= rater": entry["meets_C2_relative"], "crowd_vs_student": entry["crowd_vs_student_rho"],
                     "ENHANCE_automated": entry["enhance_automated_vs_crowd_rho"], "best_feature": best,
                     "best_feature_rho": single[best], "group_model_rho": g_rho, "all_features_rho": all_rho,
                     "group_model_vs_student": entry["group_model_vs_student_rho"], "abs_C2 (>=0.5)": entry["meets_C2_absolute"]})
    (OUT / "enhance_report.json").write_text(json.dumps(report, indent=1))
    table = pd.DataFrame(rows).round(3)
    (OUT / "enhance_summary.md").write_text(
        "# Concept validation vs ENHANCE (ISIC 2017, dermoscopy, GT masks)\n\n"
        f"Feature version {fx.FEATURE_VERSION}. Spearman rho with the mean crowd rating; group/all-feature models are "
        "5-fold cross-validated ridge regressions. rater_vs_other_raters / features_vs_other_raters score a held-out "
        "rater and the feature model against the same mean of the other raters (equal footing). Labels with rater "
        "agreement < 0.2 cannot validate anything.\n\n"
        + table.to_markdown(index=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
