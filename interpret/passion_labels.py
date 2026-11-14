"""
PASSION (MICCAI 2024) label table and subject-grouped split for the darker-skin scope detector (plan decision 2026-11-04:
darker-skin resource only, never a benign/malignant source; images stay on the laptop).

Rows: one per extracted image (<subject_id>_<k>.jpg). Subjects with pelvic involvement or no recorded body site are
excluded (decision 2026-11-04, confirmed by Michael): body site is labelled per subject, so the subject is the unit.
Label: dark = Fitzpatrick V-VI (the convention used for ISIC, MIDAS and the dev set; FST IV counts as not dark).
Split: by subject group, 60/20/20 train/val/test, stratified by FST group (I-III, IV, V, VI), seed 0, as for SCIN.
With --emb, subjects whose images are near-duplicates (DINOv2-L cosine >= 0.95, the SCIN rule) are joined first.
Only the columns the detector needs are written: no age, sex, country or body site.

    python interpret/passion_labels.py --data data/passion/PASSION_MICCAI_2024 --images data/passion/files_768 \
        [--emb classifier/output/embeddings/dinov2__passion.npz] --out splits/passion_v1.csv   (gitignored)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from interpret.scin_labels import NEAR_DUP_COS, SEED, near_duplicates

EXCLUDED_BODY_SITES = ("pelvic",)


def excluded(body_loc):
    """True for subjects left out by the 2026-11-04 decision: pelvic involvement or no recorded body site."""
    s = "" if pd.isna(body_loc) else str(body_loc).strip()
    return s == "" or any(b in [t.strip() for t in s.split(";")] for b in EXCLUDED_BODY_SITES)


def fst_group(f):
    return {5: "V", 6: "VI", 4: "IV"}.get(int(f), "I-III")


def subject_groups(subjects, pairs=()):
    """subject_id -> group id; subjects in a near-duplicate pair share a group (union-find)."""
    parent = {s: s for s in subjects}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for a, b in pairs:
        parent[find(a)] = find(b)
    return {s: find(s) for s in subjects}


def split_groups(tone, groups, seed=SEED):
    """subject_id -> train/val/test, 60/20/20 over subject groups stratified by the FST group of the group's first
    subject (the SCIN rule, scin_labels.assign_split, on plain arrays: that one fails on pandas 3 Arrow string indexes)."""
    first = {}
    for s in tone:
        first.setdefault(groups[s], tone[s])
    g, strata = np.array(list(first), dtype=object), np.array(list(first.values()), dtype=object)
    tr, rest, _, s_rest = train_test_split(g, strata, test_size=0.4, stratify=strata, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=s_rest, random_state=seed)
    to = {**{x: "train" for x in tr}, **{x: "val" for x in va}, **{x: "test" for x in te}}
    return {s: to[groups[s]] for s in tone}


def build(labels, image_ids, emb_ids=None, emb=None):
    lab = labels[~labels.body_loc.map(excluded)].copy()
    img = pd.DataFrame({"image_id": sorted(image_ids)})
    img["subject_id"] = img.image_id.str.rsplit("_", n=1).str[0]
    img = img[img.subject_id.isin(set(lab.subject_id))]
    dups = pd.DataFrame(columns=["image_a", "image_b", "cosine", "case_a", "case_b"])
    if emb is not None:
        image_case = {i: [s] for i, s in zip(img.image_id, img.subject_id)}
        dups = near_duplicates(emb_ids, emb, image_case, NEAR_DUP_COS)
    groups = subject_groups(list(lab.subject_id), zip(dups.case_a, dups.case_b))
    split = split_groups(dict(zip(lab.subject_id, lab.fitzpatrick.map(fst_group))), groups)
    lab = lab.assign(subject_group=lab.subject_id.map(groups), split=lab.subject_id.map(split),
                     dark=(lab.fitzpatrick >= 5).astype(int))
    out = img.merge(lab[["subject_id", "subject_group", "split", "fitzpatrick", "dark"]], on="subject_id")
    return out.sort_values(["split", "subject_id", "image_id"]).reset_index(drop=True), dups


def main(opts):
    labels = pd.read_csv(Path(opts.data) / "label.csv", dtype={"subject_id": str})
    ids = [p.stem for p in Path(opts.images).glob("*.jpg")]
    emb_ids = emb = None
    if opts.emb:
        z = np.load(opts.emb, allow_pickle=True)
        emb_ids, emb = z["isic_id"].astype(str), z["emb"]
    out, dups = build(labels, ids, emb_ids, emb)
    out.to_csv(opts.out, index=False)
    s = out.drop_duplicates("subject_id")
    print(f"{len(out)} images, {len(s)} subjects, {len(dups)} near-duplicate pairs across subjects")
    print(pd.crosstab(out.split, out.fitzpatrick, margins=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--emb", default="")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
