"""
Split v5 = a base split (v2 by default; v4 if MIDAS training is adopted) + melanoma-rich clinical photos in TRAIN only
(decision 2026-10-22: Derm7pt clinical photos and the Asan/SNU melanoma-vs-nevus sets; ISIC clinical overview stays
locked). Val and test rows are the base split's, so selection scores stay comparable with every earlier run.

Sources (research licences; tagged per row, never part of a commercial-safe subset):
  Derm7pt (release_v0, clinical photo of each case; CC BY-NC-ND): malignant = melanoma (all depths, in situ,
      metastasis) and BCC; benign = nevi (clark, reed/spitz, dermal, blue, congenital, combined, recurrent), SK, DF,
      lentigo, vascular, melanosis; "miscellaneous" excluded. Group = case (patient ids not released).
  Asan test + SNU subset (CAN repository): melanoma vs melanocytic nevus, Korean patients. Asan melanomas are stored at
      much higher resolution than its nevi (a known shortcut), so every Asan/SNU image is first downsampled to at most
      MAX_SIDE_CAN px and then cached like the others. Group = image (no lesion ids; near-duplicates removed by hash).
Images are cached at <= 768 px (same as the ISIC cache) as <id>.jpg in --out, ids prefixed D7_/ASAN_/SNU_ so they
cannot collide with ISIC or MIDAS ids. Near-duplicates of split-v2 images (256-bit dHash, Hamming <= 10) are dropped.

    python splits/make_split_v5.py --base isic_clinical_v2.csv --out data/extra_768
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "eval"))
from external_test import DUP_HAMMING, dhash, near_duplicates, v2_hashes  # noqa: E402

MAX_SIDE = 768
MAX_SIDE_CAN = 224   # equalizes the Asan melanoma/nevus resolution gap before caching
D7_MAP = {"basal cell carcinoma": ("Basal cell carcinoma", 1), "melanoma metastasis": ("Melanoma metastasis", 1),
          "melanoma (in situ)": ("Melanoma in situ", 1), "seborrheic keratosis": ("Seborrheic keratosis", 0),
          "dermatofibroma": ("Dermatofibroma", 0), "lentigo": ("Lentigo NOS", 0), "vascular lesion": ("Vascular lesion", 0),
          "melanosis": ("Melanosis", 0)}


def d7_label(dx):
    if dx in D7_MAP:
        return D7_MAP[dx]
    if dx.startswith("melanoma"):
        return ("Melanoma Invasive" if "mm" in dx else "Melanoma, NOS"), 1
    if "nevus" in dx:
        return "Nevus", 0
    return None, None   # miscellaneous


def resize(im, max_side):
    s = max_side / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im


def sources():
    rows = []
    d7 = ROOT / "data/derm7pt/release_v0"
    meta = pd.read_csv(d7 / "meta/meta.csv")
    for r in meta.itertuples():
        dx3, label = d7_label(r.diagnosis)
        if label is None:
            continue
        rows.append({"isic_id": f"D7_{r.case_num:04d}", "path": str(d7 / "images" / r.clinic), "label": label,
                     "group_id": f"D7_{r.case_num:04d}", "diagnosis_3": dx3, "source": "Derm7pt",
                     "copyright_license": "CC-BY-NC-ND (research)", "max_side": MAX_SIDE})
    for site, prefix in (("asantest", "ASAN"), ("snu", "SNU")):
        for cls, (dx3, label) in (("malignantmelanoma", ("Melanoma, NOS", 1)), ("melanocyticnevus", ("Nevus", 0))):
            for p in sorted((ROOT / "data/can_test" / site / cls).iterdir()):
                iid = f"{prefix}_{cls[:3]}_{p.stem}"
                rows.append({"isic_id": iid, "path": str(p), "label": label, "group_id": iid, "diagnosis_3": dx3,
                             "source": prefix.capitalize() if prefix == "ASAN" else prefix,
                             "copyright_license": "research (CAN repository)", "max_side": MAX_SIDE_CAN})
    return pd.DataFrame(rows)


def find_image(path):
    """Derm7pt meta and folders disagree on case (FCl vs FCL): resolve case-insensitively."""
    p = Path(path)
    if p.exists():
        return p
    parent = next((d for d in p.parent.parent.iterdir() if d.name.lower() == p.parent.name.lower()), None)
    return next((f for f in parent.iterdir() if f.name.lower() == p.name.lower()), None) if parent else None


def main(opts):
    out = ROOT / opts.out
    out.mkdir(parents=True, exist_ok=True)
    extra = sources()
    hashes, keep = [], []
    for r in extra.itertuples():
        src = find_image(r.path)
        if src is None:
            keep.append(False)
            hashes.append(None)
            continue
        im = Image.open(src).convert("RGB")
        im = resize(resize(im, r.max_side), MAX_SIDE)
        im.save(out / f"{r.isic_id}.jpg", quality=95)
        hashes.append(dhash(im))
        keep.append(True)
    extra = extra[keep].reset_index(drop=True)
    ref_ids, ref_h = v2_hashes(ROOT / "eval/output/external/_v2_dhash.npz")
    dup = near_duplicates([h for h, k in zip(hashes, keep) if k], ref_ids, ref_h)
    extra["v2_near_duplicate"] = dup
    # within the new sources too (Asan/SNU have repeated shots, no lesion ids)
    hs = [h for h, k in zip(hashes, keep) if k]
    bits = np.unpackbits(np.stack(hs), axis=1)
    first = {}
    for i in range(len(hs)):
        d = (bits[:i] != bits[i]).sum(1) if i else np.array([])
        close = np.flatnonzero(d <= DUP_HAMMING)
        first[i] = extra.group_id[close[0]] if len(close) else extra.group_id[i]
    extra["group_id"] = [first[i] for i in range(len(extra))]
    n_dup = int(extra.v2_near_duplicate.notna().sum())
    extra = extra[extra.v2_near_duplicate.isna()].drop(columns=["path", "max_side", "v2_near_duplicate"])
    base = pd.read_csv(HERE / opts.base)
    base["modality"] = base.get("modality", "clinical")
    extra = extra.assign(split="train", lesion_id=extra.group_id, patient_id="", diagnosis_1=np.where(extra.label == 1, "Malignant", "Benign"),
                         diagnosis_2="", attribution=extra.source, diagnosis_confirm_type="", exclusion_reason="",
                         modality="clinical")
    v5 = pd.concat([base, extra[base.columns.intersection(extra.columns).tolist()]], ignore_index=True)
    assert not set(extra.isic_id) & set(base.isic_id)
    v5.to_csv(HERE / opts.split_name, index=False)
    added = extra.groupby(["source", "label"]).size().unstack(fill_value=0)
    added["melanoma"] = extra.groupby("source").diagnosis_3.apply(lambda s: s.str.contains("Melanoma").sum())
    text = (f"# Split {opts.split_name}: {opts.base} + melanoma-rich clinical photos (train only)\n\n"
            f"Dropped as near-duplicates of split v2: {n_dup}.\n\n" + added.to_markdown() + "\n")
    (HERE / opts.split_name.replace(".csv", "_report.md")).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="isic_clinical_v2.csv")
    ap.add_argument("--split_name", default="isic_v5_clinical_plus_melanoma.csv")
    ap.add_argument("--out", default="data/extra_768")
    main(ap.parse_args())
