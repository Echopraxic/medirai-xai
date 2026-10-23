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
# Phrase bank v1 (2026-10-10): terms whose clinical reading failed validation or was never tested use plain
# image-property wording ("red tone", "image texture"), not clinical signs (erythema, scale, crust). Validation
# status per term: concepts/phrase_bank.py -> concepts/output/phrase_bank_v1.csv.
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
    "C_frac_red": ("share of red-toned areas", "high", "low"),
    "C_frac_light_brown": ("share of light-brown areas", "high", "low"),
    "C_frac_dark_brown": ("share of dark-brown areas", "high", "low"),
    "C_frac_blue_gray": ("share of blue-gray areas", "high", "low"),
    "C_frac_black": ("share of black areas", "high", "low"),
    "C_n_colors": ("color variety", "high", "low"),  # no colour count until validated (NOT_PHRASED in treeshap_top3)
    "C_color_entropy": ("color variety", "high", "low"),
    "C_color_std": ("color variation", "high", "low"),
    "C_deltaE_vs_skin": ("color contrast with surrounding skin", "high", "low"),
    "C_darkness_vs_skin": ("pigmentation relative to surrounding skin", "dark", "light"),
    "C_redness_vs_skin": ("red tone relative to surrounding skin", "redder", "less red"),
    "C_yellowness_vs_skin": ("yellow tone relative to surrounding skin", "more yellow", "less yellow"),
    "T_glcm_contrast": ("image texture contrast", "high", "low"),
    "T_glcm_homogeneity": ("image texture", "homogeneous", "heterogeneous"),
    "T_glcm_energy": ("texture uniformity", "uniform", "non-uniform"),
    "T_glcm_correlation": ("texture regularity", "regular", "irregular"),
    "T_glcm_entropy": ("texture complexity", "complex", "simple"),
    "T_lbp_entropy": ("micro-texture variety", "varied", "uniform"),
    "T_lbp_flat_fraction": ("flat micro-texture", "prevalent", "sparse"),
    "T_roughness": ("fine-scale image texture", "coarse", "fine"),   # not scale/crust (MONET rho 0.14)
    "S_eccentricity": ("elongation", "elongated", "round"),
    "S_solidity": ("compactness of the shape", "compact", "lobulated"),
    "S_extent": ("fill of the bounding box", "compact", "spread out"),
    "S_axis_ratio": ("roundness", "round", "elongated"),
}
for i in range(1, 8):
    PHRASES[f"S_hu{i}"] = (f"shape moment {i} (not clinically mapped)", "high", "low")

HIGH_PCT, LOW_PCT = 75.0, 25.0
INDEPENDENT_DOUBT_CUT = 0.34  # same rule as llm_input.INDEPENDENT_DOUBT_CUT (template imports nothing from it)
INDEPENDENT_NOTE = ("An independent model based on the measured lesion features gives this prediction less support, so it should be reviewed with caution.")


OUTLINE_GROUPS = ("A_shape", "B_", "S_")


def encoding_label(evidence: dict) -> str:
    """Code-added label for a reason whose feature the explained ResNet does not represent (probe R² < 0.3): the
    reason stays (TreeSHAP ranked it) but the reader learns it is the interpretable model's view, not the image model's."""
    if evidence.get("model_encoded", True):
        return ""
    if evidence["feature_id"].startswith(OUTLINE_GROUPS):
        return " (an outline measure the image model does not appear to use)"
    return " (a measure the image model does not appear to use)"


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
    flag = payload["uncertainty"].get("deferred")  # uncertainty-table flag, as llm_input.is_uncertain
    if bool(flag) if flag is not None else payload["uncertainty"]["entropy_of_expected"] > entropy_threshold:
        notes.append("The model is uncertain about this case and it should be reviewed by a clinician.")
        hedges.append("uncertain")
    if not payload["surrogate_agrees"]:
        notes.append("The interpretable feature model does not reproduce this prediction, so the reasons below "
                     "may not reflect what drove it.")
        hedges.append("surrogate_disagrees")
    p_pred = payload.get("feature_model_p_predicted")
    if hedges[:1] != ["uncertain"] and p_pred is not None and (1 - p_pred) > INDEPENDENT_DOUBT_CUT:
        notes.append(INDEPENDENT_NOTE)
        hedges.append("independent_check_weak")
    reasons = [{"text": describe(r, pred), "feature_id": r["feature_id"], "direction": r["direction"]}
               for r in payload["top_concepts"]]
    if any(r["direction"] == "argues_against" for r in payload["top_concepts"]):
        notes.append("The evidence is mixed: at least one of the main features argues against the prediction.")
        hedges.append("mixed_evidence")
    limitations = ("Features are computed from an automatic lesion outline on a clinical photograph; size is "
                   "relative to the frame, not in millimetres, and evolution cannot be assessed from one image.")
    return Explanation(summary, reasons, " ".join(notes), limitations, hedges)


SCOPE_NOTE = ("This image differs from the images the model was trained on, so the prediction and the reasons below "
              "are less reliable and it should be reviewed by a clinician.")
TONE_NOTE = ("The skin in this image appears darker than the skin the model was mainly validated on, and its accuracy "
             "is lower for darker skin, so the prediction and the reasons below are less reliable and it should be "
             "reviewed by a clinician.")
SHARE_SENTENCE = {
    "most": "The named features below account for most of this decision.",
    "about half": "The named features below account for about half of this decision; the rest comes from image "
                  "patterns without a clinical name.",
    "little": "The named features below account for little of this decision; most of it comes from image patterns "
              "without a clinical name.",
}


def render_llm_input(inp):
    """Template baseline on the number-free LLM input (llm_input.build): same facts the LLM gets, fixed wording.
    This is the no-LLM arm compared with every prompt strategy (T6.6)."""
    pred = inp["prediction"]
    against = any(e["direction"] == "argues_against" for e in inp["evidence"])
    summary = f"The model predicts {pred} with {inp['confidence']} confidence."
    if against:
        summary += " The evidence is mixed."
    if inp.get("named_concept_share"):
        summary += " " + SHARE_SENTENCE[inp["named_concept_share"]]
    notes = []
    if inp.get("tone_unvalidated"):
        notes.append(TONE_NOTE)
    if inp.get("in_scope") is False:
        notes.append(SCOPE_NOTE)
    if inp["uncertain"]:
        notes.append("The model is uncertain about this case and it should be reviewed by a clinician.")
    if not inp["surrogate_agrees"]:
        notes.append("The interpretable model does not reproduce this prediction, so the reasons may not reflect "
                     "what drove it.")
    if inp.get("independent_check_weak"):
        notes.append(INDEPENDENT_NOTE)
    if not inp["segmentation_reliable"]:
        notes.append("The automatic lesion outline may be inaccurate.")
    reasons = []
    for e in inp["evidence"]:
        verb = "supports" if e["direction"] == "supports" else "argues against"
        reasons.append({"feature_id": e["feature_id"], "direction": e["direction"],
                        "text": f"{e['measure'].capitalize()} is {e['finding']} ({e['relative_to_benign']}); "
                                f"this {e['strength']}ly {verb} the {pred} prediction{encoding_label(e)}."})
    return {"summary": summary, "reasons": reasons, "uncertainty_note": " ".join(notes),
            "limitations": " ".join(inp["limitations"])}
