"""
Build the versioned feature table (one row per isic_id) from images + lesion masks.

Columns: isic_id, split, label, diagnosis_3, ova_class, attribution, mask_quality (+ mask QA stats),
feature_version, then the <GROUP>_<name> feature columns. Images whose mask fails to give a usable
lesion get NaN features and extraction_error set; they stay in the table so nothing is silently dropped.

    python features/build_table.py --images uncertaintyNet-main/datasets_768/ISIC_clinical \
        --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet \
        --quality segmentation/output/qa/mask_quality.csv --out features/output/feature_table_v0.1.parquet
"""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import extract as fx

ROOT = Path(__file__).resolve().parents[1]

# One-vs-all classes (decision 2026-10-01): 5 named classes + Other; keratoacanthoma stays in Other.
OVA_MAP = {
    "Basal cell carcinoma": "BCC",
    "Nevus": "Nevus",
    "Seborrheic keratosis": "SK",
    "Squamous cell carcinoma, Invasive": "SCC",
    "Squamous cell carcinoma, NOS": "SCC",
    "Squamous cell carcinoma in situ": "SCC",
    "Melanoma Invasive": "Melanoma",
    "Melanoma in situ": "Melanoma",
    "Melanoma, NOS": "Melanoma",
    "Melanoma metastasis": "Melanoma",
}
OVA_CLASSES = ["BCC", "Nevus", "SCC", "SK", "Melanoma", "Other"]


def one(job):
    isic_id, img_path, mask_path = job
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)) > 127
        return {"isic_id": isic_id, **fx.extract(img, mask), "extraction_error": ""}
    except Exception as err:  # keep the row; the error is reported
        return {"isic_id": isic_id, "extraction_error": f"{type(err).__name__}: {err}"}


def main(opts):
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v1.csv")
    split = split[split.split != "excluded"].reset_index(drop=True)
    jobs = [(i, Path(opts.images) / f"{i}.jpg", Path(opts.masks) / f"{i}.png") for i in split.isic_id]
    with ProcessPoolExecutor(opts.workers) as ex:
        rows = list(ex.map(one, jobs, chunksize=32))
    feats = pd.DataFrame(rows)
    meta = split[["isic_id", "split", "label", "diagnosis_3", "attribution"]].copy()
    meta["ova_class"] = meta.diagnosis_3.map(OVA_MAP).fillna("Other")
    table = meta.merge(feats, on="isic_id", how="left")
    if opts.quality:
        q = pd.read_csv(opts.quality)
        keep = ["isic_id", "mask_quality", "unet_sam_dice", "unet_area_ratio", "unet_border_touch",
                "unet_boundary_uncertainty"]
        table = table.merge(q[keep], on="isic_id", how="left")
    table.insert(5, "feature_version", fx.FEATURE_VERSION)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    feature_cols = [c for c in table.columns if c.split("_")[0] in fx.CONCEPT_GROUPS and c in fx.FEATURE_DICTIONARY]
    dictionary = pd.DataFrame([{"feature": k, "group": v[0], "concept": fx.CONCEPT_GROUPS[v[0]], "units": v[1],
                                "meaning": v[2], "version": fx.FEATURE_VERSION} for k, v in fx.FEATURE_DICTIONARY.items()])
    dictionary.to_csv(out.with_name(f"feature_dictionary_{fx.FEATURE_VERSION}.csv"), index=False)
    manifest = {"feature_version": fx.FEATURE_VERSION, "work_side": fx.WORK_SIDE, "n_rows": len(table),
                "n_features": len(feature_cols), "n_errors": int((table.extraction_error.fillna("") != "").sum()),
                "masks": str(opts.masks), "images": str(opts.images),
                "split_sha256_16": hashlib.sha256((ROOT / "splits" / "isic_clinical_v1.csv").read_bytes()).hexdigest()[:16],
                "extract_py_sha256_16": hashlib.sha256(Path(fx.__file__).read_bytes()).hexdigest()[:16]}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--quality", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    main(ap.parse_args())
