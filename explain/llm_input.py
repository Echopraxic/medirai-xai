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

from template import PHRASES

LLM_INPUT_VERSION = "llm_input_v1"
PHRASE_BANK_VERSION = "phrase_bank_v0_percentile"

# percentile of benign training lesions -> relative-position phrase (upper bound exclusive)
PERCENTILE_BANDS = [
    (10, "much lower than in most benign lesions"),
    (25, "lower than in most benign lesions"),
    (75, "within the typical range of benign lesions"),
    (90, "higher than in most benign lesions"),
    (101, "much higher than in most benign lesions"),
]
CONFIDENCE_BANDS = [(0.85, "high"), (0.65, "moderate"), (0.0, "low")]  # on max(p, 1 - p)
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


def band(percentile: float) -> str:
    for upper, phrase in PERCENTILE_BANDS:
        if percentile < upper:
            return phrase
    return PERCENTILE_BANDS[-1][1]


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


def build(payload: dict, mask_quality: str, entropy_threshold: float) -> dict:
    """payload: `llm_input` from trees/treeshap_top3.py (contains numbers) -> number-free LLM input."""
    uncertain = payload["uncertainty"]["entropy_of_expected"] > entropy_threshold
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
            "relative_to_benign": band(pct),
            "direction": r["direction"],
            "strength": r["strength"],
            "clinically_mapped": mapped,
        })
    return {
        "schema_version": LLM_INPUT_VERSION,
        "phrase_bank": PHRASE_BANK_VERSION,
        "audience": payload.get("audience", "dermatologist"),
        "prediction": payload["prediction"],
        "confidence": confidence(payload["probability_malignant"]),
        "uncertain": uncertain,
        "uncertainty_reason": "the image model's predictive uncertainty is high" if uncertain else "",
        "surrogate_agrees": bool(payload["surrogate_agrees"]),
        "segmentation_reliable": mask_quality == "ok",
        "evidence": evidence,
        "limitations": LIMITATIONS,
    }


def input_hash(llm_input: dict) -> str:
    return hashlib.sha256(json.dumps(llm_input, sort_keys=True).encode()).hexdigest()[:16]


def entropy_threshold_from_val(rows) -> float:
    """C6 operating point: 75th percentile of validation entropy-of-expected (25% deferral)."""
    import numpy as np
    vals = [r["llm_input"]["uncertainty"]["entropy_of_expected"] for r in rows if r["split"] == "val"]
    return float(np.quantile(vals, 0.75))
