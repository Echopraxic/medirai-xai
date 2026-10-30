"""
SCIN label table and case-level split for the darker-skin and phone-framing detectors (S1 OOD/subgroup work).

SCIN (Google, SCIN Data Use License): ~5,000 US cases, up to 3 phone photos each. Not usable as benign/malignant
training data (mostly inflammatory, no biopsy); used here only for skin tone and photo framing. Licence terms that
bind this script's outputs: attribution to Google / the SCIN dataset and a reference to the SCIN License must travel
with any shared file, and no attempt may be made to re-identify or re-link data subjects (so demographics are only
summarised, never joined to anything outside SCIN).

Inputs (public CSVs, gs://dx-scin-public-data/dataset/): data/scin/scin_cases.csv, data/scin/scin_labels.csv.
No images are read.

Labels, all case-level (the dermatologists and raters judged the case, not each photo):
  efst          median of the gradable dermatologist Fitzpatrick labels (1-3 raters; most cases have 1);
                n_efst_raters records how many. Self-reported FST (sun response) is kept for comparison only.
  dark          1 if efst >= 4.5 (Fitzpatrick V-VI, or a 2-rater IV/V tie), 0 below, empty if no gradable label.
                Ties count as dark because the label trains a safety warning that must not miss them (decision
                2026-10-30); they are flagged in efst_tie so results can be reported without them.
  dark_agreed   1 if >= 2 raters and all of them say V-VI (a stricter positive set for checks)
  monk_us/india Monk Skin Tone 1-10 from lay raters, descriptive only
  shot_type     per image: CLOSE_UP, AT_AN_ANGLE, AT_DISTANCE (phone framing target)
  neoplasm_like any tumour-like condition in the dermatologist differential (descriptive; SCIN has no biopsy)

Split: by case group, 60/20/20 train/val/test, stratified by eFST group (I-II, III-IV, V-VI, ungradable),
seed 0. A case group joins cases that share an image file (15 image files appear in 2-3 cases in release 1,
apparently resubmissions), so the same photo can never sit in two splits. With --emb (DINOv2-L embeddings of the
SCIN images, built on Narval) it also joins cases whose images are near-duplicates (cosine >= 0.95: on inspection
the same person's lesion resubmitted as a new case; below 0.95 pairs become different people with similar rashes).
REDACTED_IMAGES are privacy-redacted photos (almost all black box, no skin to read) and are dropped, as are
WITHDRAWN_IMAGES (listed in the case table but missing from the public bucket).

Outputs:
  splits/scin_v1.csv             one row per image: image_id, case_id, split, labels above
  splits/scin_v1_near_dups.csv   image pairs from different cases joined by --emb (ids and cosine only)
  splits/scin_v1_report.md       counts by split, tone source agreement, shot type, race (summary only)

    python interpret/scin_labels.py [--emb classifier/output/embeddings/dinov2__scin.npz]
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
NEAR_DUP_COS = 0.95
# Inspected 2026-10-30: the four share one embedding (cosine 1.0) because the photo is almost entirely a black box
REDACTED_IMAGES = {"-4593817128438983108", "2542656689420019332", "4207723573736028617", "5520969433413629950"}
# Listed in scin_cases.csv but absent from the public bucket (HTTP 404, checked 2026-10-30): no pixels to embed
WITHDRAWN_IMAGES = {"-2243186711511406658"}
NEOPLASM_TERMS = ("melanoma", "basal cell", "squamous cell", "carcinoma", "nevus", "naevus", "seborrheic keratosis",
                  "actinic keratosis", "keratoacanthoma", "dermatofibroma", "lipoma", "cyst", "neoplasm", "tumor",
                  "tumour", "lentigo", "angioma", "pyogenic granuloma", "neurofibroma", "skin tag", "acrochordon")
ATTRIBUTION = ("Derived from the SCIN dataset (Google), used under the SCIN Data Use License "
               "(https://github.com/google-research-datasets/scin). Re-identification of data subjects is prohibited.")


def fst_number(s):
    """'FST5' -> 5.0; NONE_IDENTIFIED / NaN -> NaN."""
    return pd.to_numeric(s.astype(str).str.extract(r"FST([1-6])")[0], errors="coerce")


def efst_labels(labels):
    vals = []
    for i in (1, 2, 3):
        v = fst_number(labels[f"dermatologist_fitzpatrick_skin_type_label_{i}"])
        ok = labels[f"dermatologist_gradable_for_fitzpatrick_skin_type_{i}"].eq("YES")
        vals.append(v.where(ok))
    m = pd.concat(vals, axis=1)
    out = pd.DataFrame({"case_id": labels.case_id, "efst": m.median(axis=1), "n_efst_raters": m.notna().sum(axis=1)})
    out["efst_tie"] = out.efst.mod(1).eq(0.5)
    out["dark"] = np.where(out.efst.isna(), np.nan, (out.efst >= 4.5).astype(float))
    out["dark_agreed"] = ((m >= 5) | m.isna()).all(axis=1) & (out.n_efst_raters >= 2)
    return out


def tone_group(efst):
    g = pd.cut(efst, [0, 2.5, 4.5, 6], labels=["I-II", "III-IV", "V-VI"]).astype(object)
    return g.where(g.notna(), "ungradable")


def neoplasm_like(weighted):
    def names(s):
        try:
            return " ".join(ast.literal_eval(s)).lower() if isinstance(s, str) else ""
        except (ValueError, SyntaxError):
            return str(s).lower()
    text = weighted.map(names)
    return text.map(lambda t: any(k in t for k in NEOPLASM_TERMS))


def case_table(cases, labels):
    e = efst_labels(labels)
    d = cases[["case_id", "fitzpatrick_skin_type", "combined_race", "related_category"]].merge(e, on="case_id")
    lab = labels.set_index("case_id")
    d["monk_us"] = d.case_id.map(lab.monk_skin_tone_label_us.where(lab.gradable_for_monk_skin_tone_us == True))  # noqa: E712
    d["monk_india"] = d.case_id.map(lab.monk_skin_tone_label_india.where(lab.gradable_for_monk_skin_tone_india == True))  # noqa: E712
    d["sfst"] = fst_number(d.fitzpatrick_skin_type)
    d["neoplasm_like"] = d.case_id.map(neoplasm_like(lab.weighted_skin_condition_label))
    d["tone_group"] = tone_group(d.efst)
    return d.drop(columns=["fitzpatrick_skin_type"])


def case_groups(cases, extra_pairs=()):
    """Map case_id -> group id, joining cases that share any image file or appear together in extra_pairs
    (union-find)."""
    parent = {c: c for c in cases.case_id}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    seen = {}
    for i in (1, 2, 3):
        for c, path in cases[["case_id", f"image_{i}_path"]].dropna().itertuples(index=False):
            if path in seen:
                parent[find(c)] = find(seen[path])
            else:
                seen[path] = c
    for a, b in extra_pairs:
        parent[find(a)] = find(b)
    return cases.case_id.map(find)


def near_duplicates(emb_ids, emb, image_case, threshold=NEAR_DUP_COS):
    """Image pairs from different cases with embedding cosine >= threshold.

    emb_ids: image ids (str) aligned with the rows of emb; image_case: image id -> list of case ids."""
    ids, ix = np.unique(np.asarray(emb_ids).astype(str), return_index=True)
    x = np.asarray(emb, dtype=np.float32)[ix]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    rows = []
    for i0 in range(0, len(ids), 1024):
        s = x[i0:i0 + 1024] @ x.T
        for i, j in zip(*np.where(s >= threshold)):
            a, b = ids[i0 + i], ids[j]
            if a < b and a in image_case and b in image_case and set(image_case[a]).isdisjoint(image_case[b]):
                rows.append((a, b, float(s[i, j])))
    out = pd.DataFrame(rows, columns=["image_a", "image_b", "cosine"])
    out["case_a"] = out.image_a.map(lambda i: image_case[i][0])
    out["case_b"] = out.image_b.map(lambda i: image_case[i][0])
    return out.sort_values("cosine", ascending=False).reset_index(drop=True)


def assign_split(d, groups, seed=SEED):
    """Split case groups; a group's stratum is the tone group of its first case (groups are almost all size 1)."""
    g = pd.DataFrame({"case_id": d.case_id, "group": d.case_id.map(groups), "tone_group": d.tone_group})
    first = g.drop_duplicates("group").set_index("group").tone_group
    tr, rest = train_test_split(first.index, test_size=0.4, stratify=first, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=first.loc[rest], random_state=seed)
    split = pd.Series("train", index=first.index)
    split[va] = "val"
    split[te] = "test"
    return g.group.map(split)


def image_table(cases, d):
    rows = []
    for i in (1, 2, 3):
        x = cases[["case_id", f"image_{i}_path", f"image_{i}_shot_type"]].dropna(subset=[f"image_{i}_path"])
        rows.append(pd.DataFrame({"case_id": x.case_id, "image_index": i, "image_path": x[f"image_{i}_path"],
                                  "shot_type": x[f"image_{i}_shot_type"]}))
    img = pd.concat(rows, ignore_index=True)
    img.insert(0, "image_id", img.image_path.map(lambda p: Path(p).stem))
    img = img[~img.image_id.isin(REDACTED_IMAGES | WITHDRAWN_IMAGES)]
    keep = ["case_id", "case_group", "split", "efst", "n_efst_raters", "efst_tie", "dark", "dark_agreed", "tone_group", "sfst",
            "monk_us", "monk_india", "neoplasm_like"]
    return img.merge(d[keep], on="case_id").sort_values(["split", "case_id", "image_index"]).reset_index(drop=True)


def report(d, img, dups):
    lines = ["# SCIN v1 labels and split", "", ATTRIBUTION, "",
             f"{len(d)} cases, {len(img)} images in {d.case_group.nunique()} case groups (cases sharing an image "
             f"file are joined). Split by case group, stratified by eFST group, seed {SEED}.", "",
             f"Near-duplicates (DINOv2 cosine >= {NEAR_DUP_COS}, different cases): {len(dups)} image pairs, joined "
             f"into case groups before splitting" + ("" if len(dups) else " (no --emb given: not checked)") + ". "
             f"Privacy-redacted images dropped: {len(REDACTED_IMAGES)}. "
             f"Images missing from the public bucket dropped: {len(WITHDRAWN_IMAGES)}.", "",
             "## Cases by split and eFST group", "",
             pd.crosstab(d.tone_group, d.split, margins=True).to_markdown(), "",
             "## Images by split and shot type (dark = eFST V-VI)", "",
             pd.crosstab([img.shot_type, img.dark.map({1.0: "dark", 0.0: "not dark"}).fillna("ungradable")],
                         img.split, margins=True).to_markdown(), "",
             "## Dermatologist raters per case", "",
             d.n_efst_raters.value_counts().sort_index().rename("cases").to_frame().to_markdown(), "",
             f"Ties at 4.5 (counted dark): {int(d.efst_tie.sum())}. "
             f"Dark with >= 2 raters all agreeing: {int(d.dark_agreed.sum())} of "
             f"{int(((d.dark == 1) & (d.n_efst_raters >= 2)).sum())} dark cases with >= 2 raters.", "",
             "## Tone sources: eFST group vs self-reported FST", "",
             pd.crosstab(d.tone_group, tone_group(d.sfst), margins=True).to_markdown(), "",
             "## eFST group vs Monk (US raters), median and IQR", "",
             d.groupby("tone_group").monk_us.describe()[["count", "25%", "50%", "75%"]].round(1).to_markdown(), "",
             "## eFST group vs Monk (India raters), median and IQR", "",
             d.groupby("tone_group").monk_india.describe()[["count", "25%", "50%", "75%"]].round(1).to_markdown(), "",
             "## Self-reported race (summary only, never a label) by eFST group", "",
             pd.crosstab(d.combined_race.fillna("not reported"), d.tone_group).to_markdown(), "",
             f"Neoplasm-like differential (descriptive, no biopsy): {int(d.neoplasm_like.sum())} cases, "
             f"of which eFST V-VI: {int((d.neoplasm_like & (d.dark == 1)).sum())}.", ""]
    return "\n".join(lines)


def main(opts):
    src = Path(opts.data)
    cases, labels = pd.read_csv(src / "scin_cases.csv"), pd.read_csv(src / "scin_labels.csv")
    assert cases.case_id.is_unique and set(cases.case_id) == set(labels.case_id), "cases and labels do not match"
    d = case_table(cases, labels)
    dups = pd.DataFrame(columns=["image_a", "image_b", "cosine", "case_a", "case_b"])
    if opts.emb:
        z = np.load(opts.emb, allow_pickle=True)
        image_case = {}
        for i in (1, 2, 3):
            for c, path in cases[["case_id", f"image_{i}_path"]].dropna().itertuples(index=False):
                image_case.setdefault(Path(path).stem, []).append(c)
        keep = [str(i) not in REDACTED_IMAGES for i in z["isic_id"]]
        dups = near_duplicates(z["isic_id"][keep], z["emb"][keep], image_case)
    groups = dict(zip(cases.case_id, case_groups(cases, zip(dups.case_a, dups.case_b))))
    d["case_group"] = d.case_id.map(groups)
    d["split"] = assign_split(d, groups)
    img = image_table(cases, d)
    assert img.groupby("image_path").split.nunique().max() == 1, "an image file is in two splits"
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.to_csv(out, index=False)
    dups.to_csv(out.with_name(out.stem + "_near_dups.csv"), index=False)
    text = report(d, img, dups)
    out.with_name(out.stem + "_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/scin"))
    ap.add_argument("--out", default=str(ROOT / "splits/scin_v1.csv"))
    ap.add_argument("--emb", help="DINOv2 embeddings of the SCIN images (npz with isic_id = image id, emb)")
    main(ap.parse_args())
