"""
C2 concept validation on expert-labelled clinical photos (prereg S2_C2_JUDGE_PREREG_2026-10-23.md, section A).

Confirmatory pairs are fixed in PAIRS below (copied from the prereg). For each pair the candidate (a handcrafted feature or
a MONET concept score) is the continuous predictor of the concept label: AUROC with a 95% cluster-bootstrap interval,
status validated >= 0.75 / partial >= 0.60 / failed, "robust" when the interval's lower bound clears the status
threshold, and a confound check (AUROC within benign and within malignant images; the lower one is reported).
The exploratory scan (all features x all concepts, discovery on one half and replication on the other) never changes a
status. Read-only on data; writes concepts/output/c2_validation.{json,md}.

    python concepts/c2_validation.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "concepts/output"
N_BOOT = 2000
MIN_POS = 40
VALIDATED, PARTIAL = 0.75, 0.60

PAIRS = {  # (dataset, concept) -> candidates; "-" prefix = inverted (higher value means concept absent)
    ("derm7pt", "blue_whitish_veil"): ["C_frac_blue_gray", "C_deltaE_vs_skin", "M_blue_gray"],
    ("derm7pt", "pigmentation"): ["C_darkness_vs_skin", "C_color_std", "C_color_entropy", "A_color_asym_deltaE", "M_pigmented"],
    ("derm7pt", "regression_structures"): ["C_frac_white", "C_frac_blue_gray", "M_white_scar"],
    ("derm7pt", "vascular_structures"): ["C_frac_red", "C_redness_vs_skin", "M_vessels", "M_erythema"],
    ("derm7pt", "elevation"): ["T_roughness", "T_glcm_contrast", "M_nodule", "-M_flat"],
    ("skincon", "Erythema"): ["C_frac_red", "C_redness_vs_skin", "M_erythema"],
    ("skincon", "Brown(Hyperpigmentation)"): ["C_frac_dark_brown", "C_frac_light_brown", "C_darkness_vs_skin", "M_pigmented"],
    ("skincon", "Black"): ["C_frac_black", "M_dark_pigment"],
    ("skincon", "Blue"): ["C_frac_blue_gray", "M_blue_gray"],
    ("skincon", "Gray"): ["C_frac_blue_gray", "M_blue_gray"],
    ("skincon", "White(Hypopigmentation)"): ["C_frac_white", "M_white_scar"],
    ("skincon", "Yellow"): ["C_yellowness_vs_skin"],
    ("skincon", "Scale"): ["T_roughness", "T_glcm_contrast", "T_lbp_entropy", "M_scale"],
    ("skincon", "Crust"): ["C_yellowness_vs_skin", "T_roughness", "M_ulceration_crust"],
    ("skincon", "Ulcer"): ["C_frac_red", "M_ulceration_crust"],
    ("skincon", "Telangiectasia"): ["C_redness_vs_skin", "M_vessels"],
    ("skincon", "Nodule"): ["M_nodule"],
}
DERM7PT_UNTESTABLE = ["pigment_network", "streaks", "dots_and_globules"]   # dermoscopic structures
DERM7PT_BINARY = {
    "blue_whitish_veil": lambda s: s == "present",
    "pigmentation": lambda s: s != "absent",
    "regression_structures": lambda s: s != "absent",
    "vascular_structures": lambda s: s != "absent",
    "elevation": lambda s: s.isin(["palpable", "nodular"]),
}


def status(auc):
    return "validated" if auc >= VALIDATED else "partial" if auc >= PARTIAL else "failed"


def auc_or_nan(y, x):
    return float(roc_auc_score(y, x)) if len(set(y)) == 2 else float("nan")


def boot_ci(y, x, groups, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in g}
    vals = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        if len(set(y[take])) == 2:
            vals.append(roc_auc_score(y[take], x[take]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(y, x, groups, malignant, sign):
    x = sign * x
    ok = ~np.isnan(x)
    y, x, groups, malignant = y[ok], x[ok], groups[ok], malignant[ok]
    auc = auc_or_nan(y, x)
    lo, hi = boot_ci(y, x, groups)
    st = status(auc)
    thr = {"validated": VALIDATED, "partial": PARTIAL, "failed": 0.0}[st]
    strata = {}
    for name, m in (("benign", ~malignant), ("malignant", malignant)):
        strata[name] = auc_or_nan(y[m], x[m]) if m.sum() > 0 and y[m].sum() >= 10 and (~y[m]).sum() >= 10 else None
    lowest = min([v for v in strata.values() if v is not None], default=None)
    return {"auroc": auc, "ci95": [lo, hi], "status": st, "robust": bool(lo >= thr and st != "failed"),
            "n": int(len(y)), "n_pos": int(y.sum()), "auroc_within_benign": strata["benign"],
            "auroc_within_malignant": strata["malignant"],
            "diagnosis_confounded": bool(lowest is not None and lowest < PARTIAL and st != "failed")}


def load():
    d7 = pd.read_csv(ROOT / "data/derm7pt/release_v0/meta/meta.csv")
    f7 = pd.read_parquet(OUT / "derm7pt_clinic_features.parquet").merge(d7, on="case_num")
    m7 = pd.read_parquet(OUT / "monet_concepts_v1__derm7pt.parquet").rename(columns={"isic_id": "case_num"})
    f7 = f7.merge(m7, on="case_num", how="left")
    dx = f7.diagnosis.str.lower()
    f7["malignant"] = dx.str.startswith("melanoma") | dx.str.contains("basal cell")
    f7["group"] = f7.case_num.astype(str)
    test_rows = set(pd.read_csv(ROOT / "data/derm7pt/release_v0/meta/test_indexes.csv")["indexes"])   # official split: row positions in meta.csv
    pos = pd.Series(range(len(d7)), index=d7.case_num)
    f7["half"] = f7.case_num.map(lambda c: "test" if int(pos[c]) in test_rows else "train")
    ann = pd.read_csv(ROOT / "data/skincon/annotations_fitzpatrick17k.csv", index_col=0)
    ann["md5hash"] = ann.ImageID.str.replace(".jpg", "", regex=False)
    f17k = pd.read_csv(ROOT / "data/fitzpatrick17k/fitzpatrick17k.csv")[["md5hash", "three_partition_label", "fitzpatrick_scale"]]
    fs = pd.read_parquet(OUT / "skincon_f17k_features.parquet")
    ms = pd.read_parquet(OUT / "monet_concepts_v1__skincon.parquet").rename(columns={"isic_id": "md5hash"})
    fs = fs.merge(ann, on="md5hash").merge(f17k, on="md5hash").merge(ms, on="md5hash", how="left")
    fs = fs[fs["Do not consider this image"] == 0].reset_index(drop=True)
    fs["malignant"] = fs.three_partition_label == "malignant"
    fs["group"] = duplicate_groups(fs.md5hash.tolist())
    fs["half"] = np.where(pd.Series(fs.group).astype("category").cat.codes.values % 2 == 0, "train", "test")
    return f7, fs


def duplicate_groups(ids):
    """Near-duplicate groups (256-bit dHash, Hamming <= 10) over the Fitzpatrick17k images; name = first member."""
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from external_test import dhash
    hs = np.unpackbits(np.stack([dhash(Image.open(ROOT / "data/fitzpatrick17k/images_skincon_neoplastic" / f"{i}.jpg")) for i in ids]), axis=1).astype(np.float32)
    s = hs.sum(1)
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(ids)):
        d = s[i] + s - 2 * hs @ hs[i]
        for j in np.flatnonzero(d <= 10):
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([ids[find(i)] for i in range(len(ids))])


def concept_labels(df, dataset, concept):
    if dataset == "derm7pt":
        return DERM7PT_BINARY[concept](df[concept]).to_numpy()
    return (df[concept] == 1).to_numpy()


def confirmatory(f7, fs):
    rows = []
    for (dataset, concept), cands in PAIRS.items():
        df = f7 if dataset == "derm7pt" else fs
        y = concept_labels(df, dataset, concept)
        if y.sum() < MIN_POS:
            rows.append({"dataset": dataset, "concept": concept, "candidate": None, "status": "not evaluated (< 40 positives)",
                         "n_pos": int(y.sum())})
            continue
        for c in cands:
            sign, col = (-1.0, c[1:]) if c.startswith("-") else (1.0, c)
            r = evaluate(y, df[col].to_numpy(dtype=float), df.group.to_numpy(), df.malignant.to_numpy(), sign)
            rows.append({"dataset": dataset, "concept": concept, "candidate": c, "kind": "MONET" if col.startswith("M_") else "handcrafted", **r})
    return pd.DataFrame(rows)


def exploratory(f7, fs):
    out = []
    feats = [c for c in f7.columns if c[:2] in ("A_", "B_", "C_", "T_", "S_") or c.startswith("M_")]
    feats = [c for c in feats if c in fs.columns and not c.startswith("M_artifact")]
    for dataset, df, concepts in (("derm7pt", f7, list(DERM7PT_BINARY)),
                                  ("skincon", fs, [c for c in pd.read_csv(ROOT / "data/skincon/annotations_fitzpatrick17k.csv", index_col=0).columns[1:-1]])):
        for concept in concepts:
            if concept not in df.columns:
                continue
            y = concept_labels(df, dataset, concept)
            if y.sum() < MIN_POS:
                continue
            best = None
            tr, te = (df.half == "train").to_numpy(), (df.half == "test").to_numpy()
            for f in feats:
                x = df[f].to_numpy(dtype=float)
                if np.isnan(x).mean() > 0.1:
                    continue
                a_tr = auc_or_nan(y[tr][~np.isnan(x[tr])], x[tr][~np.isnan(x[tr])])
                a = max(a_tr, 1 - a_tr)
                if best is None or a > best[1]:
                    best = (f, a, 1.0 if a_tr >= 0.5 else -1.0)
            f, a_disc, sign = best
            x = sign * df[f].to_numpy(dtype=float)
            ok = ~np.isnan(x[te])
            a_rep = auc_or_nan(y[te][ok], x[te][ok])
            out.append({"dataset": dataset, "concept": concept, "n_pos": int(y.sum()), "best_feature_discovery": f,
                        "direction": "higher=present" if sign > 0 else "higher=absent", "auroc_discovery": a_disc,
                        "auroc_replication": a_rep, "replicated": bool(a_disc >= 0.65 and a_rep >= 0.65)})
    return pd.DataFrame(out)


def main():
    f7, fs = load()
    conf = confirmatory(f7, fs)
    expl = exploratory(f7, fs)
    best = conf[conf.candidate.notna()].sort_values("auroc", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c2_validation.json").write_text(json.dumps({"confirmatory": conf.to_dict(orient="records"),
        "exploratory": expl.to_dict(orient="records"), "derm7pt_untestable_on_clinical_photos": DERM7PT_UNTESTABLE,
        "n": {"derm7pt": len(f7), "skincon": len(fs)}}, indent=1, default=float))
    cols = ["dataset", "concept", "candidate", "kind", "auroc", "ci95", "status", "robust", "diagnosis_confounded", "n_pos"]
    c = conf[conf.candidate.notna()].copy()
    c["ci95"] = c.ci95.map(lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]")
    c["auroc"] = c.auroc.round(2)
    md = ["# C2 concept validation (prereg 2026-10-23, section A)", "",
          f"Derm7pt clinical photos n = {len(f7)}; SkinCon neoplastic Fitzpatrick17k images n = {len(fs)}.", "",
          "## Confirmatory pairs", "", c[cols].to_markdown(index=False), "",
          "Not evaluated (< 40 positives): " + ", ".join(conf[conf.candidate.isna()].concept) if conf.candidate.isna().any() else "",
          "", "## Exploratory scan (never changes a status)", "", expl.round(2).to_markdown(index=False)]
    (OUT / "c2_validation.md").write_text("\n".join(md), encoding="utf-8")
    print((OUT / "c2_validation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
