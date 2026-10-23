"""
Split v4 = split v2 + MRA-MIDAS phone photos (classifier sprint stage B5; decisions 2026-10-21: MIDAS phone photos
only, research licences allowed now with a commercial-safe variant at stage C).

Rows:
  every split v2 row, unchanged (`modality` = clinical)
  MIDAS phone photos (6 in / 1 ft) from splits/midas_v1.csv:  train -> train,  dev_val -> val (the pre-registered
  selection data, so each run's checkpoint is also chosen on it),  dev_test -> dev_test (never loaded in training;
  reported once per final candidate), excluded / dermoscopic / virtual -> excluded
MIDAS rows get ISIC-style `diagnosis_3` strings so the 7-class head and the diagnosis sampler treat both sources alike.
`isic_id` holds the MIDAS image id (s-prd-...), which cannot collide with ISIC ids.

    python splits/make_split_v4.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MIDAS_DIAGNOSIS = {"malignant- bcc": "Basal cell carcinoma", "malignant- melanoma": "Melanoma, NOS",
                   "malignant- scc": "Squamous cell carcinoma, NOS",
                   "malignant- sccis": "Squamous cell carcinoma in situ", "benign-melanocytic nevus": "Nevus",
                   "benign-seborrheic keratosis": "Seborrheic keratosis", "benign-dermatofibroma": "Dermatofibroma",
                   "benign-hemangioma": "Hemangioma", "benign-fibrous papule": "Fibrous papule",
                   "benign-other": "Benign, other"}
SPLIT_MAP = {"train": "train", "dev_val": "val", "dev_test": "dev_test"}


def build(v2, midas):
    v2 = v2.assign(modality="clinical")
    phone = midas.modality.str.startswith("phone") & midas.split.isin(SPLIT_MAP)
    m = midas[phone]
    rows = pd.DataFrame({
        "isic_id": m.image_id, "split": m.split.map(SPLIT_MAP), "label": m.label.astype(float),
        "group_id": m.patient_id, "lesion_id": m.lesion_id, "patient_id": m.patient_id,
        "diagnosis_1": m.label.map({0: "Benign", 1: "Malignant"}), "diagnosis_2": "",
        "diagnosis_3": m.midas_path.map(MIDAS_DIAGNOSIS), "attribution": "MRA-MIDAS (Stanford AIMI)",
        "copyright_license": m.licence, "diagnosis_confirm_type": "histopathology", "source": "MIDAS",
        "exclusion_reason": "", "modality": m.modality})
    assert rows.diagnosis_3.notna().all()
    return pd.concat([v2, rows], ignore_index=True)


if __name__ == "__main__":
    v4 = build(pd.read_csv(HERE / "isic_clinical_v2.csv"), pd.read_csv(HERE / "midas_v1.csv"))
    assert not set(v4.loc[v4.source == "MIDAS", "group_id"]) & set(v4.loc[v4.source != "MIDAS", "group_id"])
    v4.to_csv(HERE / "isic_v4_clinical_plus_midas.csv", index=False)
    t = v4[v4.split.isin(["train", "val", "test", "dev_test"])]
    summary = t.groupby(["split", "source"]).agg(images=("isic_id", "size"), malignant=("label", "mean")).round(3)
    text = "# Split v4 (v2 + MIDAS phone photos)\n\n" + summary.to_markdown() + "\n"
    (HERE / "isic_v4_clinical_plus_midas_report.md").write_text(text, encoding="utf-8")
    print(text)
