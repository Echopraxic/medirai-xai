"""
Phrase bank v1 with a validation status per term (plan T3.1-T3.3, D3), generated from code and the validation reports.

For every model feature: the words explanations use (explain/template.py PHRASES), what the feature computes
(features/extract.py FEATURE_DICTIONARY), the clinical concept it is meant to capture, the evidence for that
reading from our validation runs, and a status:

  validated    agrees with expert or reference labels at least as well as a single human rater does
               (ENHANCE, usable labels only), or Spearman rho >= 0.5 / AUROC >= 0.75 on a reference score
  partial      some agreement, below that bar (rho 0.3-0.5, AUROC 0.6-0.75, or within 0.05 of rater level)
  failed       tested against its intended concept and does not track it (rho < 0.3, AUROC < 0.6)
  untestable   the only available labels are unreliable (raters disagree with each other)
  untested     no labelled data for this concept yet
Terms never shown in explanations (trees/treeshap_top3.NOT_PHRASED) and shape moments without a clinical name are
marked as such. Wording for failed/untested terms describes the image property, not a clinical sign.

    python concepts/phrase_bank.py      # -> concepts/output/phrase_bank_v1.csv / .json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explain"))
sys.path.insert(0, str(ROOT / "features"))
sys.path.insert(0, str(ROOT / "trees"))
import extract as fx  # noqa: E402
from template import PHRASES  # noqa: E402

NOT_PHRASED = {"C_n_colors"}  # mirrors trees/treeshap_top3.NOT_PHRASED (importing it would pull in xgboost/shap)
VERSION = "phrase_bank_v1"

# feature -> (intended clinical concept, ABCD / clinical relevance, [tests]); a test is
# ("enhance", concept) | ("monet", "MONET_x~feature") | ("derm7pt", label)
INTENDED = {
    "A_shape_asym_major": ("shape asymmetry", "A", [("enhance", "asymmetry")]),
    "A_shape_asym_minor": ("shape asymmetry", "A", [("enhance", "asymmetry")]),
    "A_shape_asym_max": ("shape asymmetry", "A", [("enhance", "asymmetry")]),
    "A_color_asym_deltaE": ("asymmetric color distribution", "A", [("enhance", "asymmetry"), ("derm7pt", "pigmentation")]),
    "B_circularity": ("border irregularity", "B", [("enhance", "border")]),
    "B_convexity": ("notched border", "B", [("enhance", "border")]),
    "B_fractal_dim": ("ragged border", "B", [("enhance", "border")]),
    "B_radial_cv": ("border irregularity", "B", [("enhance", "border")]),
    "B_edge_sharpness": ("abrupt edge cut-off", "B", [("enhance", "border")]),
    "B_edge_sharpness_cv": ("variable edge definition", "B", [("enhance", "border")]),
    "C_frac_white": ("white / regression areas", "C", [("enhance", "color")]),
    "C_frac_red": ("red areas (erythema)", "C", [("enhance", "color"), ("monet", "MONET_erythema~C_frac_red"),
                                                 ("monet", "MONET_vasculature_vessels~C_frac_red")]),
    "C_frac_light_brown": ("light-brown pigment", "C", [("enhance", "color")]),
    "C_frac_dark_brown": ("dark-brown pigment", "C", [("enhance", "color"), ("monet", "MONET_pigmented~C_frac_dark_brown")]),
    "C_frac_blue_gray": ("blue-gray areas (blue-whitish veil)", "C", [("enhance", "color"), ("derm7pt", "blue_whitish_veil")]),
    "C_frac_black": ("black areas", "C", [("enhance", "color")]),
    "C_n_colors": ("number of colors (ABCD C)", "C", [("enhance", "color")]),
    "C_color_entropy": ("color variegation", "C", [("enhance", "color"), ("derm7pt", "pigmentation")]),
    "C_color_std": ("color variegation", "C", [("enhance", "color"), ("derm7pt", "pigmentation")]),
    "C_deltaE_vs_skin": ("contrast with surrounding skin / pigmentation", "C",
                         [("monet", "MONET_pigmented~C_deltaE_vs_skin"), ("derm7pt", "blue_whitish_veil")]),
    "C_darkness_vs_skin": ("pigmentation", "C", [("monet", "MONET_pigmented~C_darkness_vs_skin")]),
    "C_redness_vs_skin": ("erythema", "C", [("enhance", "color"), ("monet", "MONET_erythema~C_redness_vs_skin")]),
    "C_yellowness_vs_skin": ("yellow crust / scale", "C", [("enhance", "color"), ("monet", "MONET_ulceration_crust~C_yellowness_vs_skin")]),
    "T_glcm_contrast": ("surface relief / scale", "surface", [("monet", "MONET_ulceration_crust~T_glcm_contrast"), ("derm7pt", "elevation")]),
    "T_roughness": ("surface roughness / scale", "surface", [("monet", "MONET_ulceration_crust~T_roughness"), ("derm7pt", "elevation")]),
}
RANK = {"validated": 4, "partial": 3, "failed": 2, "untestable": 1, "untested": 0}


def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def latest_vlm():
    hits = sorted(ROOT.glob("concepts/output/vlm_*/eval.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(hits[-1].read_text(encoding="utf-8")) if hits else None


def judge(test, feature, enh, monet, derm):
    kind, key = test
    if kind == "enhance" and enh:
        c = enh["concepts"][key]
        rho, rater = c["single_feature_rho"].get(feature), c["crowd_rater_vs_others_rho"]
        if rho is None:
            return None
        text = f"ENHANCE {key} (dermoscopy crowd): rho {rho:.2f} vs single rater {rater:.2f}"
        if not c["labels_usable"]:
            return "untestable", text + " — raters disagree, labels unusable"
        status = "validated" if rho >= rater else "partial" if rho >= max(0.2, rater - 0.05) else "failed"
        return status, text
    if kind == "monet" and monet and key in monet["pairs"]:
        rho = monet["pairs"][key]
        status = "validated" if rho >= 0.5 else "partial" if rho >= 0.3 else "failed"
        return status, f"MILK10k {key.split('~')[0].replace('MONET_', 'MONET ')} (clinical photos): rho {rho:.2f}"
    if kind == "derm7pt" and derm:
        auc = derm.get(key, {}).get(feature)
        if auc is None:
            return None
        status = "validated" if auc >= 0.75 else "partial" if auc >= 0.6 else "failed"
        return status, f"Derm7pt expert '{key.replace('_', ' ')}' (clinical photos): AUROC {auc:.2f}"
    return None


def build():
    enh = load("concepts/output/enhance_report.json")
    monet = load("concepts/output/monet_report.json")
    vlm = latest_vlm()
    derm = vlm.get("our_features_vs_expert_auroc") if vlm else None
    rows = []
    for fid, (group, units, meaning) in fx.FEATURE_DICTIONARY.items():
        measure, high, low = PHRASES.get(fid, (fid, "high", "low"))
        concept, relevance, tests = INTENDED.get(fid, (None, fx.CONCEPT_GROUPS.get(group, group), []))
        results = [r for r in (judge(t, fid, enh, monet, derm) for t in tests) if r]
        status = max((r[0] for r in results), key=RANK.get) if results else "untested"
        if fid.startswith("S_hu"):
            shown = "only as 'a shape pattern without a clinical name'"
        elif fid in NOT_PHRASED:
            shown = "no (model feature only)"
        else:
            shown = "yes"
        rows.append({"feature_id": fid, "concept_group": fx.CONCEPT_GROUPS.get(group, group),
                     "intended_clinical_concept": concept or "—", "clinical_relevance": relevance,
                     "explanation_wording": f"{measure}: '{high}' (high) / '{low}' (low)" if shown == "yes" else "—",
                     "what_is_computed": meaning, "units": units, "status": status,
                     "evidence": "; ".join(r[1] for r in results) or "no labelled data yet",
                     "shown_in_explanations": shown, "version": VERSION})
    return pd.DataFrame(rows)


def main():
    df = build()
    out = ROOT / "concepts/output"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{VERSION}.csv", index=False)
    (out / f"{VERSION}.json").write_text(df.to_json(orient="records", indent=1), encoding="utf-8")
    print(df.groupby("status").size().to_string())
    print(df[["feature_id", "status", "shown_in_explanations"]].to_string(index=False))


if __name__ == "__main__":
    main()
