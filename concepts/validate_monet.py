"""
Clinical-domain concept check (plan T3.3): our features vs the MONET concept scores that MILK10k ships for
the same clinical close-ups (data/milk10k/supplements/training_input.csv).

MONET (Kim et al., Nat Med 2024) is an automated image-text concept annotator, not an expert rater, so this
is a weak check. Its value is the domain: these are our own clinical photos, not dermoscopy. A concept whose
features do not track the MONET score must not be phrased with that clinical term (e.g. lesion redness is
not "erythema", texture is not "scale/crust").

    python concepts/validate_monet.py --features features/output/feature_table_v0.2.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {  # MONET concept -> our candidate features
    "MONET_pigmented": ["C_darkness_vs_skin", "C_deltaE_vs_skin", "C_frac_dark_brown"],
    "MONET_erythema": ["C_redness_vs_skin", "C_frac_red"],
    "MONET_vasculature_vessels": ["C_redness_vs_skin", "C_frac_red"],
    "MONET_ulceration_crust": ["T_roughness", "T_glcm_contrast", "C_yellowness_vs_skin"],
}
USABLE = 0.4  # a feature may carry the MONET concept's clinical name only above this rho


def main(opts):
    monet = pd.read_csv(ROOT / "data/milk10k/supplements/training_input.csv")
    monet = monet[monet.image_type == "clinical: close-up"]
    table = pd.read_parquet(opts.features)
    df = table.merge(monet, on="isic_id")
    df = df[(df.extraction_error.fillna("") == "") & (df.mask_quality != "fail")]
    report = {"n": len(df), "features": opts.features, "usable_threshold": USABLE, "pairs": {}}
    rows = []
    for concept, feats in PAIRS.items():
        for f in feats:
            r = float(spearmanr(df[concept], df[f]).statistic)
            report["pairs"][f"{concept}~{f}"] = r
            rows.append({"MONET concept": concept, "feature": f, "spearman": round(r, 3), "may use clinical name": r >= USABLE})
    out = ROOT / "concepts/output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "monet_report.json").write_text(json.dumps(report, indent=1))
    text = (f"# Clinical-domain check vs MONET (MILK10k clinical close-ups, n={len(df)})\n\n"
            + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    (out / "monet_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "features/output/feature_table_v0.2.parquet"))
    main(ap.parse_args())
