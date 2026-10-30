"""
Feature table v1.1 = v1.0 + the MONET concepts that passed C2 (decision 2026-10-30, C1 explanation source).

A concept enters only if it is validated in one C2 dataset and not failed in the other (concepts/output/
c2_validation.md, confirmatory rows): dark pigment (SkinCon 'Black' 0.94), ulceration/crust (SkinCon 'Ulcer'
0.85, 'Crust' 0.77) and vessels (SkinCon 'Telangiectasia' 0.79, Derm7pt vascular structures 0.67 partial).
M_pigmented and M_white_scar are validated on SkinCon but failed on Derm7pt, so they stay out.

Each concept is renamed into the clinical group it describes, so trees/train_trees.py picks it up by prefix and
trees/treeshap_top3.py sums its SHAP with that group's handcrafted features. Scores are MONET's
concept-minus-reference cosine on the whole image (concepts/monet_concepts.py), not mask-based.

    python features/add_monet.py   ->  features/output/feature_table_v1.1.parquet
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.1"
BASE = ROOT / "features/output/feature_table_v1.0.parquet"
MONET = ROOT / "concepts/output/monet_concepts_v1.parquet"
# new feature id -> (MONET column, C2 candidate evidence, what it computes)
MONET_FEATURES = {
    "C_monet_dark_pigment": ("M_dark_pigment", "black pigmentation, dark brown lesion"),
    "C_monet_vessels": ("M_vessels", "telangiectasia, arborizing vessels"),
    "T_monet_ulceration_crust": ("M_ulceration_crust", "ulceration, crust"),
}


def add_monet(table, monet):
    cols = {src: fid for fid, (src, _) in MONET_FEATURES.items()}
    m = monet[["isic_id", *cols]].rename(columns=cols)
    out = table.merge(m, on="isic_id", how="left", validate="one_to_one")
    assert out[list(cols.values())].notna().all().all(), "MONET score missing for a split-v2 image"
    out["feature_version"] = VERSION
    return out


def main():
    out = add_monet(pd.read_parquet(BASE), pd.read_parquet(MONET))
    path = ROOT / f"features/output/feature_table_{VERSION}.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} rows, +{len(MONET_FEATURES)} MONET features -> {path}")


if __name__ == "__main__":
    main()
