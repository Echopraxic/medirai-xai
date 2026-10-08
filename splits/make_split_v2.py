"""
Derive splits/isic_clinical_v2.csv from v1 without re-splitting (decision 2026-10-07).

v2 = v1 with the Memorial Sloan Kettering (MSKCC) contribution marked split="excluded". Every other image
keeps its v1 assignment, so v1 and v2 results are directly comparable on the shared images.

Why MSKCC is excluded: its 1,241 "clinical: close-up" images come from 73 patients, are 99.5% benign and
97% labelled by single-contributor clinical assessment (no histopathology, almost all without a specific
diagnosis). Visually they are contact/dermoscopy-style images (round vignette, polarized cast) and many show
no clearly visible lesion. They are a different modality and population from the MILK/UFES clinical photos,
and on v1 the features identify them among benign lesions with AUROC 0.97 (color 0.94, texture 0.82).

v2 also carries per-image provenance so licence and label quality can be filtered later:
`source`, `copyright_license`, `diagnosis_confirm_type`, `exclusion_reason`.

    python splits/make_split_v2.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXCLUDED_SOURCES = {"Memorial Sloan Kettering Cancer Center":
                    "MSKCC: dermoscopy-like contact images, 97% unbiopsied benign, many without a visible lesion"}
SOURCE_SHORT = {"MILK study team": "MILK", "Memorial Sloan Kettering Cancer Center": "MSKCC",
                "Hospital Italiano de Buenos Aires": "HIBA", "Anonymous": "Anonymous"}


def short_source(attribution: str) -> str:
    if attribution in SOURCE_SHORT:
        return SOURCE_SHORT[attribution]
    return "UFES" if "UFES" in str(attribution) else "other"


def build_v2(v1: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    v2 = v1.merge(meta[["isic_id", "copyright_license", "diagnosis_confirm_type"]], on="isic_id", how="left")
    v2["source"] = v2.attribution.map(short_source)
    v2["exclusion_reason"] = ""
    v2.loc[v2.split == "excluded", "exclusion_reason"] = "diagnosis_1 not Benign/Malignant (v1 rule)"
    for attribution, reason in EXCLUDED_SOURCES.items():
        drop = (v2.attribution == attribution) & (v2.split != "excluded")
        v2.loc[drop, "split"] = "excluded"
        v2.loc[drop, "exclusion_reason"] = reason
    return v2


def report(v2: pd.DataFrame) -> str:
    used = v2[v2.split != "excluded"]
    lines = ["# isic_clinical_v2", "",
             "Derived from v1 by `splits/make_split_v2.py`; no image changes split, MSKCC is excluded.", "",
             "## Images per split and label", "", pd.crosstab(used.split, used.label, margins=True).to_markdown(), "",
             "## Source x split", "", pd.crosstab(used.source, used.split, margins=True).to_markdown(), "",
             "## Licence", "", used.copyright_license.value_counts().to_markdown(), "",
             "## Label confirmation (benign only)", "",
             used[used.label == 0].diagnosis_confirm_type.value_counts().to_markdown(), "",
             "## Exclusions", "", v2.exclusion_reason.replace("", "(used)").value_counts().to_markdown(), ""]
    return "\n".join(lines)


def main():
    v1 = pd.read_csv(HERE / "isic_clinical_v1.csv")
    meta = pd.read_csv(HERE / "isic_clinical_closeup_metadata.csv")
    v2 = build_v2(v1, meta)
    v2.to_csv(HERE / "isic_clinical_v2.csv", index=False)
    text = report(v2)
    (HERE / "isic_clinical_v2_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
