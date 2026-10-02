"""
No-LLM template baseline (plan T4.4): structured explanation input -> fixed sentences.

Input: the `llm_input` payload from trees/treeshap_top3.py (prediction, probability, uncertainty, surrogate
agreement, top-3 concepts with feature_id/direction/strength/benign percentile). Output follows the planned
LLM output schema: {summary, reasons[{text, feature_id, direction}], uncertainty_note, limitations}, plus
`hedges` (which caution statements were emitted) for the C5/C6 checks.

Phrases describe feature values relative to benign training lesions (percentiles) because the WS3
thresholds that map values to clinical grades are not validated yet.
"""
from dataclasses import dataclass

# feature -> (what is measured, wording when high, wording when low)
PHRASES = {
    "A_shape_asym_major": ("shape symmetry across the long axis", "asymmetric", "symmetric"),
    "A_shape_asym_minor": ("shape symmetry across the short axis", "asymmetric", "symmetric"),
    "A_shape_asym_max": ("shape symmetry", "asymmetric", "symmetric"),
    "A_color_asym_deltaE": ("color distribution between halves", "uneven", "even"),
    "B_circularity": ("border outline", "round and regular", "irregular"),
    "B_convexity": ("border contour", "smooth", "notched or indented"),
    "B_fractal_dim": ("border complexity", "ragged", "smooth"),
    "B_radial_cv": ("border outline", "irregular", "regular"),
    "B_edge_sharpness": ("edge transition to surrounding skin", "abrupt", "gradual"),
    "B_edge_sharpness_cv": ("edge definition along the border", "variable", "uniform"),
    "C_frac_white": ("share of white/pale areas", "high", "low"),
    "C_frac_red": ("share of red areas", "high", "low"),
    "C_frac_light_brown": ("share of light-brown areas", "high", "low"),
    "C_frac_dark_brown": ("share of dark-brown areas", "high", "low"),
    "C_frac_blue_gray": ("share of blue-gray areas", "high", "low"),
    "C_frac_black": ("share of black areas", "high", "low"),
    "C_n_colors": ("number of distinct colors", "high", "low"),
    "C_color_entropy": ("color variety", "high", "low"),
    "C_color_std": ("color variation", "high", "low"),
    "C_deltaE_vs_skin": ("color contrast with surrounding skin", "high", "low"),
    "C_darkness_vs_skin": ("pigmentation relative to surrounding skin", "dark", "light"),
    "C_redness_vs_skin": ("redness relative to surrounding skin", "marked", "slight"),
    "C_yellowness_vs_skin": ("yellow tone relative to surrounding skin", "marked", "slight"),
    "D_feret_rel": ("size in the frame (not calibrated to mm)", "large", "small"),
    "D_area_rel": ("area in the frame (not calibrated to mm)", "large", "small"),
    "T_glcm_contrast": ("surface texture contrast", "high", "low"),
    "T_glcm_homogeneity": ("surface texture", "smooth", "heterogeneous"),
    "T_glcm_energy": ("texture uniformity", "uniform", "non-uniform"),
    "T_glcm_correlation": ("texture regularity", "regular", "irregular"),
    "T_glcm_entropy": ("texture complexity", "complex", "simple"),
    "T_lbp_entropy": ("micro-texture variety", "varied", "uniform"),
    "T_lbp_flat_fraction": ("flat micro-texture", "prevalent", "sparse"),
    "T_roughness": ("surface roughness (scale/crust)", "rough", "smooth"),
    "S_eccentricity": ("elongation", "elongated", "round"),
    "S_solidity": ("compactness of the shape", "compact", "lobulated"),
    "S_extent": ("fill of the bounding box", "compact", "spread out"),
    "S_axis_ratio": ("roundness", "round", "elongated"),
}
for i in range(1, 8):
    PHRASES[f"S_hu{i}"] = (f"shape moment {i} (not clinically mapped)", "high", "low")

HIGH_PCT, LOW_PCT = 75.0, 25.0


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class Explanation:
    summary: str
    reasons: list
    uncertainty_note: str
    limitations: str
    hedges: list

    def as_dict(self):
        return self.__dict__.copy()

    def text(self):
        parts = [self.summary] + ([self.uncertainty_note] if self.uncertainty_note else [])
        parts += [f"{i + 1}. {r['text']}" for i, r in enumerate(self.reasons)] + [self.limitations]
        return "\n".join(parts)


def describe(reason, prediction):
    what, high, low = PHRASES.get(reason["feature_id"], (reason["feature_id"], "high", "low"))
    pct = reason["benign_percentile"]
    if pct >= HIGH_PCT:
        value = f"{high} (higher than {pct:.0f}% of benign training lesions)"
    elif pct <= LOW_PCT:
        value = f"{low} (lower than {100 - pct:.0f}% of benign training lesions)"
    else:
        value = f"within the usual range of benign training lesions ({ordinal(round(pct))} percentile)"
    verb = "supports" if reason["direction"] == "supports" else "argues against"
    return f"{reason['concept'].capitalize()}: {what} is {value}; this {reason['strength']}ly {verb} the {prediction} prediction."


def render(payload, entropy_threshold):
    pred = payload["prediction"]
    p = payload["probability_malignant"]
    hedges = []
    summary = f"The model predicts {pred} (probability of malignancy {p:.2f})."
    notes = []
    if payload["uncertainty"]["entropy_of_expected"] > entropy_threshold:
        notes.append("The model is uncertain about this case and it should be reviewed by a clinician.")
        hedges.append("uncertain")
    if not payload["surrogate_agrees"]:
        notes.append("The interpretable feature model does not reproduce this prediction, so the reasons below "
                     "may not reflect what drove it.")
        hedges.append("surrogate_disagrees")
    reasons = [{"text": describe(r, pred), "feature_id": r["feature_id"], "direction": r["direction"]}
               for r in payload["top_concepts"]]
    if any(r["direction"] == "argues_against" for r in payload["top_concepts"]):
        notes.append("The evidence is mixed: at least one of the main features argues against the prediction.")
        hedges.append("mixed_evidence")
    limitations = ("Features are computed from an automatic lesion outline on a clinical photograph; size is "
                   "relative to the frame, not in millimetres, and evolution cannot be assessed from one image.")
    return Explanation(summary, reasons, " ".join(notes), limitations, hedges)
