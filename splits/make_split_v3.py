"""
Derive splits/isic_v3_clinical_plus_derm.csv: split v2 unchanged + biopsy-confirmed ISIC dermoscopic images as extra
TRAINING data (WS9, 2026-10-15).

Rules (val and test stay exactly the v2 clinical images, so every v2 number stays comparable):
  - every v2 row keeps its split; new rows get `modality` = "dermoscopic" (v2 rows: "clinical")
  - a dermoscopic image whose lesion_id or patient_id belongs to a v2 val/test/excluded group is excluded
    (paired MILK10k dermoscopy of a test lesion would leak the test lesion into training)
  - a dermoscopic image of a v2 TRAIN lesion/patient joins train with that group_id
  - other dermoscopic images: diagnosis_1 Benign/Malignant only (Indeterminate excluded); grouped by patient_id, else
    lesion_id, else the image; all go to train
  - MSKCC dermoscopy is kept (the v2 MSKCC exclusion was about contact-style "clinical" images, not dermoscopy);
    `source` lets a run drop it

    python splits/fetch_isic_metadata.py            # writes splits/sources/isic_dermoscopic_histo.csv
    python splits/make_split_v3.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "isic_v3_clinical_plus_derm.csv"


def short_source(attribution):
    a = str(attribution)
    for key, name in [("MILK", "MILK"), ("Memorial Sloan", "MSKCC"), ("Barcelona", "BCN"), ("Italiano", "HIBA"),
                      ("Sydney", "Sydney"), ("Queensland", "UQ"), ("Imperial", "Imperial"), ("Basel", "Basel")]:
        if key in a:
            return name
    return "Anonymous" if a == "Anonymous" else "other"


def build_v3(v2: pd.DataFrame, derm: pd.DataFrame) -> pd.DataFrame:
    v2 = v2.assign(modality="clinical")
    derm = derm[~derm.isic_id.isin(v2.isic_id)].copy()
    split_of = {}
    for key in ["lesion_id", "patient_id"]:
        g = v2.dropna(subset=[key]).groupby(key)
        split_of[key] = (g.split.agg(lambda s: "train" if set(s) == {"train"} else "held"), g.group_id.first())
    derm["split"], derm["group_id"], derm["exclusion_reason"] = "train", None, ""
    for key in ["lesion_id", "patient_id"]:
        status, gid = split_of[key]
        st = derm[key].map(status)
        held = st == "held"
        derm.loc[held, ["split", "exclusion_reason"]] = ["excluded", f"dermoscopy of a v2 val/test/excluded {key}"]
        joins = (st == "train") & derm.group_id.isna()
        derm.loc[joins, "group_id"] = derm.loc[joins, key].map(gid)
    bad = ~derm.diagnosis_1.isin(["Benign", "Malignant"]) & (derm.split != "excluded")
    derm.loc[bad, ["split", "exclusion_reason"]] = ["excluded", "diagnosis_1 not Benign/Malignant"]
    new = derm.group_id.isna()
    derm.loc[new, "group_id"] = ("D_" + derm.loc[new, "patient_id"].fillna(derm.loc[new, "lesion_id"])
                                 .fillna(derm.loc[new, "isic_id"]).astype(str))
    derm["label"] = derm.diagnosis_1.map({"Benign": 0.0, "Malignant": 1.0})
    derm["source"] = derm.attribution.map(short_source)
    derm["modality"] = "dermoscopic"
    cols = [c for c in v2.columns if c in derm.columns]
    return pd.concat([v2, derm[cols]], ignore_index=True)


def report(v3: pd.DataFrame) -> str:
    used = v3[v3.split != "excluded"]
    return "\n".join([
        "# isic_v3_clinical_plus_derm", "", "Built by `splits/make_split_v3.py`. Val/test are the v2 clinical images.", "",
        "## Images per split, modality and label", "", pd.crosstab([used.split, used.modality], used.label, margins=True).to_markdown(), "",
        "## Dermoscopic train images by source", "", used[used.modality == "dermoscopic"].source.value_counts().to_markdown(), "",
        "## Licence (used)", "", pd.crosstab(used.copyright_license, used.modality).to_markdown(), "",
        "## Exclusions", "", v3.exclusion_reason.fillna("").replace("", "(used)").value_counts().to_markdown(), ""])


def main():
    v2 = pd.read_csv(HERE / "isic_clinical_v2.csv")
    derm = pd.read_csv(HERE / "sources" / "isic_dermoscopic_histo.csv")
    v3 = build_v3(v2, derm)
    assert v3.isic_id.is_unique
    held = v3[(v3.modality == "dermoscopic") & (v3.split == "train")]
    for key in ["lesion_id", "patient_id"]:
        evalset = set(v2.loc[v2.split.isin(["val", "test"]), key].dropna())
        assert not held[key].isin(evalset).any(), f"leak via {key}"
    v3.to_csv(OUT, index=False)
    (HERE / "isic_v3_clinical_plus_derm_report.md").write_text(report(v3), encoding="utf-8")
    print(report(v3))


if __name__ == "__main__":
    main()
