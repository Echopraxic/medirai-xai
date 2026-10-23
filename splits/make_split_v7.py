"""
Split v7, the darker-skin dev set (DSD) and the tone pairs (prereg 2026-10-23, sections 1 and 3). Runs on Narval,
where the images are.

  DSD    splits/f17k_dsd_v1.csv: Fitzpatrick17k (full Kaggle mirror, ids = md5hash) with FST 4-6 in the skin-cancer task
         (mapping below, fixed in the prereg). Column `dsd` = FST 5-6 (the selection set); FST 4 rows only feed the
         secondary DSD-IV report. split = "dsd". Atlas labels: relative comparisons only.
  v7     splits/isic_v7_v6_plus_mendeley.csv = v6 + Mendeley originals (FST I-II) + their I2SB counterparts (FST V-VI)
         in train only; group = pair; column `synthetic` (1 = generated image). Val/test rows are v6's.
  pairs  splits/mendeley_pairs_v1.csv (orig_id, synth_id, label): the consistency arm S4.

Images are cached at <= 768 px as <id>.jpg into --image_dir (the training jobs' ISIC_clinical folder), ids F17K_<md5>,
MDO_<name> (original) and MDS_<name> (synthetic), so they cannot collide with ISIC/MIDAS/Derm7pt/Asan ids.

De-duplication: 256-bit dHash, Hamming <= 10 (eval/external_test.py):
  - F17k images that duplicate each other share a group (the bootstrap unit); so do Mendeley repeat shots;
  - DSD rows that duplicate any v6 image (any split) or any Mendeley image are dropped;
  - Mendeley pairs where either image duplicates a v6 val/test image, a locked-set image or a DSD image are dropped.

    python splits/make_split_v7.py --f17k_dir <dir with fitzpatrick17k.csv> --mendeley_dir <unzipped Mendeley set> \
        --image_dir $DATA/ISIC_clinical --locked_dirs <DDI files_768> <DDI-2 files_768> <MCR-SL images> \
        --locked_hashes eval/output/external/_overview_dhash.npz
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
from external_test import DUP_HAMMING, dhash  # noqa: E402

MAX_SIDE = 768
IMG_EXT = {".jpg", ".jpeg", ".png"}
F17K_LICENSE = "CC-BY-NC-SA-3.0 (Fitzpatrick17k; research)"
MENDELEY_LICENSE = "CC-BY-4.0 package (Kim et al. 2025); upstream SD-260/DermQuest terms unclear (research)"

# Fitzpatrick17k label -> (diagnosis_3, label); prereg 2026-10-23 section 1. Benign = every F17k benign label except
# prurigo nodularis; actinic keratosis, kaposi sarcoma and mycosis fungoides are outside the task.
F17K_MALIGNANT = {"melanoma": "Melanoma, NOS", "malignant melanoma": "Melanoma, NOS",
                  "superficial spreading melanoma ssm": "Melanoma, superficial spreading",
                  "lentigo maligna": "Melanoma in situ, lentigo maligna type",
                  "basal cell carcinoma": "Basal cell carcinoma", "basal cell carcinoma morpheiform": "Basal cell carcinoma",
                  "solid cystic basal cell carcinoma": "Basal cell carcinoma",
                  "squamous cell carcinoma": "Squamous cell carcinoma, NOS"}
F17K_EXCLUDED = {"prurigo nodularis", "actinic keratosis", "kaposi sarcoma", "mycosis fungoides"}
MENDELEY_MAP = {"MM": ("Melanoma, NOS", 1), "BCC": ("Basal cell carcinoma", 1), "SCC": ("Squamous cell carcinoma, NOS", 1),
                "NV": ("Nevus", 0), "SK": ("Seborrheic keratosis", 0)}


def f17k_task(df):
    """Rows of the Fitzpatrick17k csv in the task, with diagnosis_3 and label (FST >= 4)."""
    keep = df[(df.fitzpatrick_scale >= 4) & ~df.label.isin(F17K_EXCLUDED)
              & ((df.three_partition_label == "benign") | df.label.isin(F17K_MALIGNANT))].copy()
    keep["diagnosis_3"] = [F17K_MALIGNANT.get(x, x) for x in keep.label]
    keep["y"] = keep.label.isin(F17K_MALIGNANT).astype(int)
    return keep


def bits(hashes):
    return np.unpackbits(np.stack(hashes), axis=1).astype(np.float32)


def min_distance(q, ref, chunk=2048):
    """Smallest Hamming distance from each row of q to any row of ref (both 0/1 float matrices)."""
    if len(ref) == 0:
        return np.full(len(q), 256)
    out = np.empty(len(q))
    rs = ref.sum(1)
    for i in range(0, len(q), chunk):
        a = q[i:i + chunk]
        d = a.sum(1)[:, None] + rs[None, :] - 2 * a @ ref.T
        out[i:i + chunk] = d.min(1)
    return out


def groups_within(b, names, max_dist=DUP_HAMMING):
    """Union-find over near-duplicate pairs inside one set: group name = the first member's name."""
    parent = list(range(len(b)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    s = b.sum(1)
    for i in range(0, len(b), 2048):
        d = s[i:i + 2048, None] + s[None, :] - 2 * b[i:i + 2048] @ b.T
        for a, c in zip(*np.nonzero(d <= max_dist)):
            ra, rc = find(i + a), find(c)
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
    return [names[find(i)] for i in range(len(b))]


def cache(src, dest):
    """Resize to <= MAX_SIDE, save as JPEG, return the dHash of the cached copy (the image the models see)."""
    im = Image.open(src).convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dest, quality=95)
    return dhash(im)


def hash_folder(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
    return [dhash(Image.open(p)) for p in files]


def mendeley_files(root):
    """{(kind, stem): path} with kind 'orig' for original-*/ogirinal-* folders and 'synth' for synthetic-* folders."""
    out = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        top = p.relative_to(root).parts[0].lower()
        kind = "synth" if top.startswith("synth") else "orig" if top.startswith(("orig", "ogir")) else None
        if kind:
            out[(kind, p.stem)] = p
    return out


def main(opts):
    img_dir = Path(opts.image_dir)
    v6 = pd.read_csv(HERE / opts.base)
    v6_used = v6[v6.split != "excluded"]
    print(f"hashing {len(v6_used)} v6 images", flush=True)
    v6_h = {i: dhash(Image.open(img_dir / f"{i}.jpg")) for i in v6_used.isic_id}
    v6_all = bits(list(v6_h.values()))
    v6_eval = bits([v6_h[i] for i in v6_used[v6_used.split != "train"].isic_id])
    locked = []
    for d in opts.locked_dirs:
        locked += hash_folder(d)
    if opts.locked_hashes:
        locked += list(np.load(ROOT / opts.locked_hashes)["h"])
    locked = bits(locked)
    print(f"locked-set hashes: {len(locked)}", flush=True)

    # ---- Fitzpatrick17k -> DSD
    meta = pd.read_csv(Path(opts.f17k_dir) / "fitzpatrick17k.csv")
    task = f17k_task(meta)
    files = {p.stem: p for p in Path(opts.f17k_dir).rglob("*.jpg")}
    task = task[task.md5hash.isin(files)].reset_index(drop=True)
    n_missing = int((f17k_task(meta).md5hash.isin(files) == False).sum())  # noqa: E712
    task["isic_id"] = "F17K_" + task.md5hash
    f_h = [cache(files[m], img_dir / f"{i}.jpg") for m, i in zip(task.md5hash, task.isic_id)]
    f_b = bits(f_h)
    task["group_id"] = groups_within(f_b, list(task.isic_id))

    # ---- Mendeley pairs
    md = pd.read_csv(Path(opts.mendeley_dir) / "metadata.csv").rename(columns={"disease label": "disease"})
    mf = mendeley_files(opts.mendeley_dir)
    md["stem"] = md.filename.astype(str).str.rsplit(".", n=1).str[0]
    md = md[[("orig", s) in mf and ("synth", s) in mf for s in md.stem]].reset_index(drop=True)
    n_md_unpaired = int(len(pd.read_csv(Path(opts.mendeley_dir) / "metadata.csv")) - len(md))
    o_h = [cache(mf[("orig", s)], img_dir / f"MDO_{s}.jpg") for s in md.stem]
    s_h = [cache(mf[("synth", s)], img_dir / f"MDS_{s}.jpg") for s in md.stem]
    o_b, s_b = bits(o_h), bits(s_h)

    # ---- de-duplication (order fixed in the prereg: DSD first, then Mendeley against the evaluation data)
    f_dup = (min_distance(f_b, v6_all) <= DUP_HAMMING) | (min_distance(f_b, np.vstack([o_b, s_b])) <= DUP_HAMMING)
    dsd = task[~f_dup].reset_index(drop=True)
    dsd_b = f_b[~f_dup]
    evals = np.vstack([v6_eval, locked, dsd_b])
    m_dup = (min_distance(o_b, evals) <= DUP_HAMMING) | (min_distance(s_b, evals) <= DUP_HAMMING)
    md, o_b = md[~m_dup].reset_index(drop=True), o_b[~m_dup]
    md["group_id"] = groups_within(o_b, ["MDP_" + s for s in md.stem])

    # ---- write DSD
    out_dsd = pd.DataFrame({"isic_id": dsd.isic_id, "split": "dsd", "label": dsd.y, "group_id": dsd.group_id,
                            "lesion_id": dsd.group_id, "patient_id": "", "diagnosis_1": np.where(dsd.y == 1, "Malignant", "Benign"),
                            "diagnosis_3": dsd.diagnosis_3, "f17k_label": dsd.label, "fitzpatrick": dsd.fitzpatrick_scale,
                            "dsd": dsd.fitzpatrick_scale >= 5, "source": "F17k", "copyright_license": F17K_LICENSE,
                            "modality": "clinical"})
    out_dsd.to_csv(HERE / "f17k_dsd_v1.csv", index=False)

    # ---- write v7 + pairs
    rows = []
    for r in md.itertuples():
        dx3, y = MENDELEY_MAP[r.disease]
        for prefix, syn in (("MDO_", 0), ("MDS_", 1)):
            rows.append({"isic_id": prefix + r.stem, "split": "train", "label": float(y), "group_id": r.group_id,
                         "lesion_id": r.group_id, "patient_id": "", "diagnosis_1": "Malignant" if y else "Benign",
                         "diagnosis_2": "", "diagnosis_3": dx3, "attribution": "Kim et al. 2025 (SD-260)",
                         "copyright_license": MENDELEY_LICENSE, "diagnosis_confirm_type": "atlas",
                         "source": "Mendeley-synthetic" if syn else "Mendeley-original", "exclusion_reason": "",
                         "modality": "clinical", "synthetic": syn})
    extra = pd.DataFrame(rows)
    v7 = pd.concat([v6.assign(synthetic=0), extra], ignore_index=True)
    assert not set(extra.isic_id) & set(v6.isic_id) and v7.isic_id.is_unique
    v7.to_csv(HERE / "isic_v7_v6_plus_mendeley.csv", index=False)
    pairs = pd.DataFrame({"orig_id": "MDO_" + md.stem, "synth_id": "MDS_" + md.stem,
                          "label": [MENDELEY_MAP[x][1] for x in md.disease], "group_id": md.group_id})
    pairs.to_csv(HERE / "mendeley_pairs_v1.csv", index=False)

    # ---- reports (counts after de-duplication are reported, not tuned)
    by = out_dsd.groupby(["dsd", "label"]).size().unstack(fill_value=0)
    mel = out_dsd[out_dsd.diagnosis_3.str.contains("Melanoma")].groupby("dsd").size()
    text = (f"# Darker-skin dev set f17k_dsd_v1.csv (prereg 2026-10-23)\n\n"
            f"F17k task rows with FST >= 4: {len(task) + n_missing} ({n_missing} images missing from the mirror). "
            f"Dropped as near-duplicates of v6 or Mendeley images: {int(f_dup.sum())}. "
            f"Duplicate groups: {out_dsd.group_id.nunique()} for {len(out_dsd)} images.\n\n"
            f"Rows (dsd = FST 5-6; False = FST 4, DSD-IV only):\n\n{by.to_markdown()}\n\nMelanomas: {mel.to_dict()}\n")
    (HERE / "f17k_dsd_v1_report.md").write_text(text, encoding="utf-8")
    ext = extra.groupby(["source", "diagnosis_3"]).size().unstack(fill_value=0)
    text2 = (f"# Split isic_v7_v6_plus_mendeley.csv: {opts.base} + Mendeley light/dark pairs (train only)\n\n"
             f"Metadata rows without both images: {n_md_unpaired}. Pairs dropped as near-duplicates of v6 val/test, "
             f"locked or DSD images: {int(m_dup.sum())}. Pairs kept: {len(md)} in {md.group_id.nunique()} groups.\n\n"
             f"{ext.to_markdown()}\n")
    (HERE / "isic_v7_v6_plus_mendeley_report.md").write_text(text2, encoding="utf-8")
    print(text + "\n" + text2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="isic_v6_v4_plus_melanoma.csv")
    ap.add_argument("--f17k_dir", required=True)
    ap.add_argument("--mendeley_dir", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--locked_dirs", nargs="*", default=[])
    ap.add_argument("--locked_hashes", default="")
    main(ap.parse_args())
