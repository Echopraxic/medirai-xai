"""
Patient-grouped split of MRA-MIDAS (Stanford AIMI, Redivis aimi.mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer
v1.0) for the classifier sprint (decision 2026-10-21): MIDAS is training and model-selection data, never a locked test.

Split: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED) grouped by patient (`midas_record_id`),
stratified on the benign/malignant label; folds 0-2 -> train (60%), 3 -> dev_val (20%, model selection with the
internal val), 4 -> dev_test (20%, reported once per final candidate). Every image of a patient (all lesions, all
distances, dermoscopy included) lands in one split.

Labels (consistent with ISIC diagnosis_1, which split v2 trains on):
  malignant  bcc, melanoma, scc, sccis
  benign     melanocytic nevus, seborrheic keratosis, dermatofibroma, hemangioma, fibrous papule, other benign
  excluded   ak and "possible re-excision" melanocytic lesions (ISIC "Indeterminate"), malignant- other,
             non-neoplastic, controls without pathology, rows whose image file is missing from the release
Rows keep `modality` (phone_6in / phone_1ft / dermoscopic / virtual); evaluation uses the phone photos.
Licence: Stanford research use agreement (non-commercial); tagged `licence=restricted_nc`.

    python splits/make_midas_split.py --release data/midas/release_midas.csv --images data/midas/files_768
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from make_isic_clinical_split import SEED  # also enforces the scikit-learn pin

HERE = Path(__file__).resolve().parent
TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS = (0, 1, 2), (3,), (4,)
MIDAS_LABEL = {"malignant- bcc": 1, "malignant- melanoma": 1, "malignant- scc": 1, "malignant- sccis": 1,
               "benign-melanocytic nevus": 0, "benign-seborrheic keratosis": 0, "benign-dermatofibroma": 0,
               "benign-hemangioma": 0, "benign-fibrous papule": 0, "benign-other": 0}
MODALITY = {"6in": "phone_6in", "1ft": "phone_1ft", "dscope": "dermoscopic", "n/a - virtual": "virtual"}


def exclusion_reason(row, have_image):
    if not have_image:
        return "image file missing from the release"
    if row.midas_path in MIDAS_LABEL:
        return ""
    if pd.isna(row.midas_path) or str(row.midas_path).strip() in ("", "0"):
        return "no pathology (control or missing)"
    return f"label not benign/malignant in ISIC terms: {row.midas_path}"


def build(release, image_dir):
    df = release.copy()
    have = {p.stem for p in Path(image_dir).glob("*.jpg")}
    df["image_id"] = df.midas_file_name.map(lambda f: Path(f).stem)
    df["exclusion_reason"] = [exclusion_reason(r, r.image_id in have) for r in df.itertuples()]
    df["label"] = df.midas_path.map(MIDAS_LABEL)
    df["patient_id"] = "MIDAS_" + df.midas_record_id.astype(str)
    loc = df.midas_location.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    df["lesion_id"] = df.patient_id + "|" + loc
    df["modality"] = df.midas_distance.map(MODALITY).fillna("unknown")
    df["split"] = "excluded"
    usable = df.exclusion_reason == ""
    # one row per patient for fold assignment: stratify on whether the patient has any malignant lesion
    pat = df[usable].groupby("patient_id").label.max().reset_index()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_of = {}
    for k, (_, idx) in enumerate(folds.split(pat, pat.label, groups=pat.patient_id)):
        fold_of.update({p: k for p in pat.patient_id.iloc[idx]})
    name = {**{f: "train" for f in TRAIN_FOLDS}, **{f: "dev_val" for f in VAL_FOLDS}, **{f: "dev_test" for f in TEST_FOLDS}}
    df.loc[usable, "split"] = df.loc[usable, "patient_id"].map(fold_of).map(name)
    df["licence"] = "restricted_nc"
    cols = ["image_id", "midas_file_name", "split", "label", "patient_id", "lesion_id", "modality", "midas_path",
            "midas_fitzpatrick", "length__mm_", "width__mm_", "licence", "exclusion_reason"]
    return df[cols].sort_values(["split", "patient_id", "image_id"]).reset_index(drop=True)


def report(df):
    use = df[df.split != "excluded"]
    lines = ["# MIDAS split v1 (patient-grouped)", "",
             f"{len(df)} release rows; {len(use)} usable images of {use.patient_id.nunique()} patients; "
             f"{(df.split == 'excluded').sum()} excluded.", "",
             "| split | patients | images | phone images | malignant share (phone) |", "|---|---|---|---|---|"]
    for s in ("train", "dev_val", "dev_test"):
        d = use[use.split == s]
        ph = d[d.modality.str.startswith("phone")]
        lines.append(f"| {s} | {d.patient_id.nunique()} | {len(d)} | {len(ph)} | {ph.label.mean():.3f} |")
    lines += ["", "Exclusions:", ""] + [f"- {k}: {v}" for k, v in
                                        df[df.split == "excluded"].exclusion_reason.value_counts().items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=str(HERE.parent / "data/midas/release_midas.csv"))
    ap.add_argument("--images", default=str(HERE.parent / "data/midas/files_768"))
    ap.add_argument("--out", default=str(HERE / "midas_v1.csv"))
    opts = ap.parse_args()
    split = build(pd.read_csv(opts.release), opts.images)
    assert not (split[split.split != "excluded"].groupby("patient_id").split.nunique() > 1).any()  # no patient leak
    split.to_csv(opts.out, index=False)
    Path(opts.out).with_name("midas_v1_report.md").write_text(report(split), encoding="utf-8")
    print(report(split))
