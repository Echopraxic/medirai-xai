"""
Splits for the classifier sprint part 3 (prereg S2_CLASSIFIER_PREREG_2026-10-30.md): the cleaned darker-skin dev set,
two new-data arms and the darker-skin learning curve. Runs on Narval, where the images are.

  DSD v2  splits/f17k_dsd_v2.csv: the FST 5-6 rows of f17k_dsd_v1.csv that Fitzpatrick17k-C keeps (Abhishek et al.,
          Zenodo 12739457, CC-BY-4.0: duplicates and outliers removed) and that it does not flag "3 Wrongly labelled";
          plus `fold` (0-2, by duplicate group, stratified by label, seed 0) for the learning curve. Selection only.
  v8f     isic_v8f_v6_plus_f17k.csv = v6 + Fitzpatrick17k-C task rows with FST 1-4 (mapping of make_split_v7.py,
          applied to the F17k-C labels), train only.
  v8s     isic_v8s_v6_plus_sd198.csv = v6 + SD-198 (DermQuest atlas, Kaggle longngzzz/sd-198) tumour classes, train only.
  LC      isic_v8f_lc_k{k}_n{n}.csv = v8f + the DSD v2 rows of folds k+1..k+n (mod 3) in train; fold k is held out.

De-duplication (256-bit dHash, Hamming <= 10, as eval/external_test.py): new training images that duplicate any v6
val/test image, a locked-set image or a DSD v2 image are dropped; duplicates inside a source share a group.
Fitzpatrick17k-C is itself de-duplicated across FST labels, so DSD v2 and the F17k FST 1-4 rows share no image.

    python splits/make_split_v8.py --f17k_dir <F17k mirror> --f17k_c <Fitzpatrick17k-C.csv> --sd198_dir <unzipped> \
        --image_dir $DATA/ISIC_clinical --locked_dirs <...> --locked_hashes eval/output/external/_overview_dhash.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from make_split_v7 import (DUP_HAMMING, F17K_EXCLUDED, F17K_LICENSE, F17K_MALIGNANT, IMG_EXT, bits, cache, dhash,
                           groups_within, hash_folder, min_distance)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 0
SD198_LICENSE = "unknown (Kaggle mirror of SD-198 / DermQuest; research only)"
# SD-198 class -> diagnosis_3 (prereg part 3, fixed from the class list before any training). Malignant follows split
# v2 (keratoacanthoma and SCC in situ are malignant there); actinic/arsenical keratosis, metastases, lymphomas and
# hamartomas (epidermal / sebaceous / Becker nevus) are outside the task.
SD198_MALIGNANT = {"Basal_Cell_Carcinoma": "Basal cell carcinoma", "Malignant_Melanoma": "Melanoma, NOS",
                   "Lentigo_Maligna_Melanoma": "Melanoma, lentigo maligna", "Bowens_Disease": "Squamous cell carcinoma in situ",
                   "Keratoacanthoma": "Keratoacanthoma"}
SD198_BENIGN = {**{c: "Nevus" for c in ("Blue_Nevus", "Compound_Nevus", "Congenital_Nevus", "Dysplastic_Nevus", "Halo_Nevus",
                                        "Junction_Nevus", "Nevus_Incipiens", "Nail_Nevus", "Nevus_Spilus")},
                "Seborrheic_Keratosis": "Seborrheic keratosis", "Benign_Keratosis": "Seborrheic keratosis",
                "Solar_Lentigo": "Solar lentigo", "Dermatofibroma": "Dermatofibroma", "Angioma": "Hemangioma",
                "Strawberry_Hemangioma": "Hemangioma", "Pyogenic_Granuloma": "Pyogenic granuloma", "Skin_Tag": "Acrochordon",
                "Neurofibroma": "Neurofibroma", "Epidermoid_Cyst": "Cyst", "Trichilemmal_Cyst": "Cyst", "Lipoma": "Lipoma",
                "Sebaceous_Gland_Hyperplasia": "Sebaceous hyperplasia", "Eccrine_Poroma": "Poroma",
                "Apocrine_Hydrocystoma": "Hidrocystoma"}


def f17k_c_task(c, fst_range):
    """Fitzpatrick17k-C rows in the task (make_split_v7 mapping on the corrected labels) with fst in fst_range."""
    keep = c[c.fst.isin(fst_range) & ~c.label.isin(F17K_EXCLUDED)
             & ((c.three_partition_label == "benign") | c.label.isin(F17K_MALIGNANT))].copy()
    keep["diagnosis_3"] = [F17K_MALIGNANT.get(x, x) for x in keep.label]
    keep["y"] = keep.label.isin(F17K_MALIGNANT).astype(int)
    return keep


def dsd_v2(dsd_v1, c):
    """FST 5-6 DSD rows kept by F17k-C and not flagged wrongly labelled, with a 3-fold column by duplicate group."""
    keep = set(c.loc[c.qc.fillna("") != "3 Wrongly labelled", "md5hash"])
    d = dsd_v1[dsd_v1.dsd & dsd_v1.isic_id.str.removeprefix("F17K_").isin(keep)].reset_index(drop=True)
    d["fold"] = -1
    for k, (_, te) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(d, d.label, d.group_id)):
        d.loc[te, "fold"] = k
    return d


def train_rows(ids, y, dx3, groups, source, attribution, licence, confirm):
    return pd.DataFrame({"isic_id": ids, "split": "train", "label": np.asarray(y, float), "group_id": groups,
                         "lesion_id": groups, "patient_id": "", "diagnosis_1": np.where(np.asarray(y) == 1, "Malignant", "Benign"),
                         "diagnosis_2": "", "diagnosis_3": dx3, "attribution": attribution, "copyright_license": licence,
                         "diagnosis_confirm_type": confirm, "source": source, "exclusion_reason": "", "modality": "clinical"})


def lc_split(v8f, dsd, k, n):
    """Learning-curve split: v8f + the DSD v2 rows of folds k+1..k+n (mod 3) as training rows; fold k stays held out."""
    add = dsd[dsd.fold.isin([(k + i) % 3 for i in range(1, n + 1)])]
    rows = train_rows(add.isic_id, add.label, add.diagnosis_3, add.group_id, "F17k-DSD", "Fitzpatrick17k", F17K_LICENSE, "atlas")
    return pd.concat([v8f, rows], ignore_index=True)


def main(opts):
    img_dir = Path(opts.image_dir)
    v6 = pd.read_csv(HERE / opts.base)
    used = v6[v6.split != "excluded"]
    v6_eval = bits([dhash(Image.open(img_dir / f"{i}.jpg")) for i in used[used.split != "train"].isic_id])
    locked = [h for d in opts.locked_dirs for h in hash_folder(d)]
    if opts.locked_hashes:
        locked += list(np.load(ROOT / opts.locked_hashes)["h"])
    c = pd.read_csv(opts.f17k_c)
    dsd = dsd_v2(pd.read_csv(HERE / "f17k_dsd_v1.csv"), c)
    dsd_b = bits([dhash(Image.open(img_dir / f"{i}.jpg")) for i in dsd.isic_id])
    evals = np.vstack([v6_eval, bits(locked), dsd_b])

    # ---- F17k-C FST 1-4
    files = {p.stem: p for p in Path(opts.f17k_dir).rglob("*.jpg")}
    f = f17k_c_task(c, range(1, 5))
    n_f_missing = int((~f.md5hash.isin(files)).sum())
    f = f[f.md5hash.isin(files) & ~("F17K_" + f.md5hash).isin(dsd.isic_id)].reset_index(drop=True)
    f["isic_id"] = "F17K_" + f.md5hash
    f_b = bits([cache(files[m], img_dir / f"{i}.jpg") for m, i in zip(f.md5hash, f.isic_id)])
    f_dup = min_distance(f_b, evals) <= DUP_HAMMING
    f, f_b = f[~f_dup].reset_index(drop=True), f_b[~f_dup]
    f_rows = train_rows(f.isic_id, f.y, f.diagnosis_3, groups_within(f_b, list(f.isic_id)), "F17k-C",
                        "Fitzpatrick17k (Groh et al. 2021), cleaned by Fitzpatrick17k-C", F17K_LICENSE, "atlas")
    f_rows["fitzpatrick"] = f.fst.values

    # ---- SD-198
    sd = [(p, cls) for cls in {**SD198_MALIGNANT, **SD198_BENIGN} for p in sorted((Path(opts.sd198_dir) / cls).glob("*"))
          if p.suffix.lower() in IMG_EXT]
    s_ids = [f"SD198_{cls}_{p.stem}" for p, cls in sd]
    s_b = bits([cache(p, img_dir / f"{i}.jpg") for (p, _), i in zip(sd, s_ids)])
    s_dup = min_distance(s_b, evals) <= DUP_HAMMING
    keep = np.where(~s_dup)[0]
    s_cls = [sd[i][1] for i in keep]
    s_rows = train_rows([s_ids[i] for i in keep], [int(x in SD198_MALIGNANT) for x in s_cls],
                        [{**SD198_MALIGNANT, **SD198_BENIGN}[x] for x in s_cls], groups_within(s_b[keep], [s_ids[i] for i in keep]),
                        "SD-198", "SD-198 (Sun et al. 2016, DermQuest)", SD198_LICENSE, "atlas")

    v8f = pd.concat([v6, f_rows], ignore_index=True)
    v8s = pd.concat([v6, s_rows], ignore_index=True)
    assert v8f.isic_id.is_unique and v8s.isic_id.is_unique
    dsd.to_csv(HERE / "f17k_dsd_v2.csv", index=False)
    v8f.to_csv(HERE / "isic_v8f_v6_plus_f17k.csv", index=False)
    v8s.to_csv(HERE / "isic_v8s_v6_plus_sd198.csv", index=False)
    for k in range(3):
        for n in (1, 2):
            lc_split(v8f, dsd, k, n).to_csv(HERE / f"isic_v8f_lc_k{k}_n{n}.csv", index=False)

    text = (f"# Split v8 and DSD v2 (prereg 2026-10-30)\n\n"
            f"DSD v2 (FST 5-6 kept by Fitzpatrick17k-C): {len(dsd)} images, {dsd.group_id.nunique()} groups\n\n"
            f"{dsd.groupby(['fold', 'label']).size().unstack(fill_value=0).to_markdown()}\n\n"
            f"F17k-C FST 1-4 task rows missing from the mirror: {n_f_missing}; dropped as duplicates of evaluation "
            f"data: {int(f_dup.sum())}; kept: {len(f_rows)}\n\n"
            f"{f_rows.groupby(['fitzpatrick', 'label']).size().unstack(fill_value=0).to_markdown()}\n\n"
            f"SD-198 task images: {len(sd)}; dropped as duplicates of evaluation data: {int(s_dup.sum())}; kept: {len(s_rows)}\n\n"
            f"{s_rows.groupby(['diagnosis_3', 'label']).size().unstack(fill_value=0).to_markdown()}\n")
    (HERE / "isic_v8_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="isic_v6_v4_plus_melanoma.csv")
    ap.add_argument("--f17k_dir", required=True)
    ap.add_argument("--f17k_c", required=True)
    ap.add_argument("--sd198_dir", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--locked_dirs", nargs="*", default=[])
    ap.add_argument("--locked_hashes", default="")
    main(ap.parse_args())
