"""
Attribution encodings for the paper's RQ2 (S2 paper proposal §4.2): how the same TreeSHAP evidence reaches the LLM.

Two axes, so granularity is separated from format:
  F-raw     feature level  raw SHAP + raw feature value     {"feature_id": "T_glcm_contrast", "feature_value": 0.412,
                                                             "shap_toward_prediction": 0.183}
  C-raw     concept level  grouped SHAP number              {"feature_id": ..., "concept": "texture/surface",
                                                             "shap_toward_prediction": 0.21}
  C-bin     concept level  direction + strength words       {"feature_id": ..., "concept": ..., "direction": "supports",
                                                             "strength": "strong"}
  C-phrase  concept level  clinical phrase + direction +    the pipeline's number-free llm_input evidence
                           strength (pipeline default)       (explain/llm_input.py)

Every encoding cites the same top-3 concept representatives except F-raw, which cites the top-3 individual
features by |SHAP| (the "paste the SHAP table" baseline of prior narration work). For each encoding, `encode`
returns:
  prompt_input    what the LLM sees (prediction + evidence in that format; never images, metadata or identifiers)
  verify_input    an llm_input-shaped dict for explain/verifier.py, describing the items this encoding cites
  allowed_numbers numbers present in the prompt (copying them is not fabrication; inventing others is)
The summary, caveats and limitations are built in code for every encoding (hybrid mode), so the encodings differ
only in how the evidence is presented.
"""
import copy

import llm_input as li

ENCODINGS = ("F-raw", "C-raw", "C-bin", "C-phrase")
GROUP_NAMES = {"A": "asymmetry", "B": "border", "C": "color", "D": "diameter", "T": "texture/surface", "S": "shape"}


def _numbers(items, keys):
    return sorted({str(it[k]) for it in items for k in keys if k in it})


def encode(payload, mask_quality, entropy_threshold, encoding):
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; choose from {ENCODINGS}")
    if encoding == "F-raw":
        if not payload.get("top_features"):
            raise ValueError("F-raw needs `top_features` in the payload (trees/treeshap_top3.py, 2026-10-10 or later)")
        # re-express the top features as concept-like items so the shared verifier can check them
        pseudo = copy.deepcopy(payload)
        pseudo["top_concepts"] = [{"concept": GROUP_NAMES[f["group"]], "group": f["group"], "feature_id": f["feature_id"],
                                   "direction": "supports" if f["shap"] > 0 else "argues_against", "strength": "moderate",
                                   "grouped_shap": f["shap"], "feature_value": f["feature_value"],
                                   "benign_percentile": f["benign_percentile"]} for f in payload["top_features"]]
        verify_input = li.build(pseudo, mask_quality, entropy_threshold)
        evidence = [{"feature_id": f["feature_id"], "feature_value": f["feature_value"],
                     "shap_toward_prediction": f["shap"]} for f in payload["top_features"]]
        allowed = _numbers(evidence, ("feature_value", "shap_toward_prediction"))
    else:
        verify_input = li.build(payload, mask_quality, entropy_threshold)
        items = payload["top_concepts"]
        if encoding == "C-raw":
            evidence = [{"feature_id": r["feature_id"], "concept": r["concept"],
                         "shap_toward_prediction": r["grouped_shap"]} for r in items]
            allowed = _numbers(evidence, ("shap_toward_prediction",))
        elif encoding == "C-bin":
            evidence = [{"feature_id": r["feature_id"], "concept": r["concept"], "direction": r["direction"],
                         "strength": r["strength"]} for r in items]
            allowed = []
        else:  # C-phrase
            evidence = verify_input["evidence"]
            allowed = []
    prompt_input = {"audience": verify_input["audience"], "prediction": verify_input["prediction"],
                    "evidence": evidence}
    return prompt_input, verify_input, allowed
