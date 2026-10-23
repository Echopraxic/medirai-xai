"""
LLM input and output schemas (plan T4.2 / T4.3) and phrase bank v0.

The LLM never sees a number. Prior MedirAI work found Llama could not turn feature values into explanations,
so every value is converted into a categorical finding here, before the LLM:
  - feature value  -> finding word ("irregular", "dark") + position relative to benign training lesions
  - probability    -> confidence category
  - entropy        -> uncertainty flag (validation 75th percentile = the C6 operating point)
Because the input has no digits, any digit in the output is an unsupported claim (checked by verifier.py).

Phrase bank v0 is percentile-based and UNVALIDATED (WS3 replaces it with data-calibrated grades).
Shape moments (S_hu*) are marked clinically_mapped=False: they may be cited only as
"a shape pattern without a clinical name".
"""
import hashlib
import json

from template import PHRASES, encoding_label  # noqa: F401 (re-exported)

LLM_INPUT_VERSION = "llm_input_v2"  # v2: independent_check_weak
PHRASE_BANK_VERSION = "phrase_bank_v1_neutral"  # v1: neutral wording for unvalidated terms

# percentile of benign training lesions -> relative-position phrase (upper bound exclusive)
PERCENTILE_BANDS = [
    (10, "much lower than in most benign lesions"),
    (25, "lower than in most benign lesions"),
    (75, "within the typical range of benign lesions"),
    (90, "higher than in most benign lesions"),
    (101, "much higher than in most benign lesions"),
]
CONFIDENCE_BANDS = [(0.85, "high"), (0.65, "moderate"), (0.0, "low")]  # on max(p, 1 - p)
# Confident predictions whose independent feature tree (ground-truth binary tree) gives the predicted class less
# than this probability get a caveat. Fit on validation confident predictions (top 20% of doubt) by
# eval/error_signals.py, 2026-10-10: on test it flags 23% of confident predictions, catches 42% of confident errors,
# error rate 25% flagged vs 10.5% unflagged. Refit whenever the explained ResNet50 changes.
INDEPENDENT_DOUBT_CUT = 0.34
LIMITATIONS = [
    "Features are measured from an automatic lesion outline on a single clinical photograph.",
    "Lesion size cannot be given in millimetres and evolution over time cannot be assessed from one image.",
    "The reasons come from an interpretable model that approximates the image classifier.",
]

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "reasons", "uncertainty_note", "limitations"],
    "properties": {
        "summary": {"type": "string"},
        "reasons": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["feature_id", "direction", "text"],
                      "properties": {"feature_id": {"type": "string"},
                                     "direction": {"type": "string", "enum": ["supports", "argues_against"]},
                                     "text": {"type": "string"}}}},
        "uncertainty_note": {"type": "string"},
        "limitations": {"type": "string"},
    },
}


# hybrid mode: the LLM writes only the reason sentences; summary, caveats and limitations are built in code
# (template.render_llm_input), so the uncertainty, surrogate, outline and mixed-evidence statements hold by
# construction. Motivated by the 2026-10-07 Rorqual run: LLMs dropped or invented these caveats.
REASONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasons"],
    "properties": {"reasons": OUTPUT_SCHEMA["properties"]["reasons"]},
}

def band(percentile: float) -> str:
    for upper, phrase in PERCENTILE_BANDS:
        if percentile < upper:
            return phrase
    return PERCENTILE_BANDS[-1][1]


# Comparatives for finding words; anything else becomes "more <word>".
COMPARATIVE = {"dark": "darker", "light": "lighter", "coarse": "coarser", "fine": "finer", "smooth": "smoother",
               "simple": "simpler", "round": "rounder", "sparse": "sparser", "redder": "redder",
               "less red": "less red", "more yellow": "more yellow", "less yellow": "less yellow"}


def relative(percentile: float, word: str) -> str:
    """Position among benign training lesions, worded along the FINDING, not the raw score: an asymmetry score below
    most benign lesions is "more symmetric than most benign lesions", never "lower" next to "symmetric"."""
    if 25 <= percentile < 75 or word == "typical":
        return "within the typical range of benign lesions"
    much = "much " if percentile < 10 or percentile >= 90 else ""
    if word in ("high", "low"):  # the measure is the quantity itself ("share of red-toned areas")
        return f"{much}{'higher' if word == 'high' else 'lower'} than in most benign lesions"
    return f"{much}{COMPARATIVE.get(word, 'more ' + word)} than most benign lesions"


def finding(feature_id: str, percentile: float) -> str:
    _, high, low = PHRASES.get(feature_id, (feature_id, "high", "low"))
    if percentile >= 75:
        return high
    if percentile < 25:
        return low
    return "typical"


def confidence(p_malignant: float) -> str:
    conf = max(p_malignant, 1 - p_malignant)
    return next(name for cut, name in CONFIDENCE_BANDS if conf >= cut)


def is_uncertain(payload: dict, entropy_threshold: float) -> bool:
    """Per-case deferral flag from the uncertainty table (explain/attach_deferral.py: high entropy OR model
    disagreement) when attached; otherwise the validation entropy threshold."""
    flag = payload["uncertainty"].get("deferred")
    return bool(flag) if flag is not None else payload["uncertainty"]["entropy_of_expected"] > entropy_threshold


def build(payload: dict, mask_quality: str, entropy_threshold: float) -> dict:
    """payload: `llm_input` from trees/treeshap_top3.py (contains numbers) -> number-free LLM input."""
    uncertain = is_uncertain(payload, entropy_threshold)
    p_pred = payload.get("feature_model_p_predicted")  # absent in pre-v2 inputs -> no caveat
    independent_weak = (not uncertain) and p_pred is not None and (1 - p_pred) > INDEPENDENT_DOUBT_CUT
    evidence = []
    for r in payload["top_concepts"]:
        fid = r["feature_id"]
        measure = PHRASES.get(fid, (fid,))[0]
        mapped = not fid.startswith("S_hu")
        pct = r["benign_percentile"]
        if mapped:
            word = finding(fid, pct)
        else:
            measure = "a shape pattern without a clinical name"
            word = "typical" if 25 <= pct < 75 else "unusual"
        evidence.append({
            "feature_id": fid,
            "concept": r["concept"],
            "measure": measure,
            "finding": word,
            "relative_to_benign": relative(pct, word),
            "direction": r["direction"],
            "strength": r["strength"],
            "clinically_mapped": mapped,
            **({"model_encoded": r["model_encoded"]} if "model_encoded" in r else {}),
        })
    extra = {}  # optional explanation features (interpret/explanation_features.py); absent keys keep old input hashes
    if payload.get("in_scope") is not None:
        extra["in_scope"] = bool(payload["in_scope"])
    if payload.get("named_concept_share"):
        extra["named_concept_share"] = payload["named_concept_share"]
    if payload.get("tone_unvalidated"):
        extra["tone_unvalidated"] = True
    return {
        "schema_version": LLM_INPUT_VERSION,
        "phrase_bank": PHRASE_BANK_VERSION,
        "audience": payload.get("audience", "dermatologist"),
        "prediction": payload["prediction"],
        "confidence": confidence(payload["probability_malignant"]),
        "uncertain": uncertain,
        "uncertainty_reason": "the image model's predictive uncertainty is high" if uncertain else "",
        "surrogate_agrees": bool(payload["surrogate_agrees"]),
        "independent_check_weak": independent_weak,
        "segmentation_reliable": mask_quality == "ok",
        "evidence": evidence,
        "limitations": LIMITATIONS,
        **extra,
    }


def input_hash(llm_input: dict) -> str:
    return hashlib.sha256(json.dumps(llm_input, sort_keys=True).encode()).hexdigest()[:16]


def entropy_threshold_from_val(rows) -> float:
    """C6 operating point: 75th percentile of validation entropy-of-expected (25% deferral)."""
    import numpy as np
    vals = [r["llm_input"]["uncertainty"]["entropy_of_expected"] for r in rows if r["split"] == "val"]
    return float(np.quantile(vals, 0.75))
