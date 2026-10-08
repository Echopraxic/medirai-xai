"""
Rule-based verifier for explanations (plan T6.1). Works on any output in the shared schema
{summary, reasons[{feature_id, direction, text}], uncertainty_note, limitations}: the template baseline and
every LLM strategy are checked with the same rules.

Checks, per explanation:
  parse_ok / schema_ok      the output is one JSON object with the required keys
  grounding (C3)            share of reasons whose feature_id is one of the supplied evidence items
  coverage                  share of supplied evidence items cited
  direction (C4)            share of grounded reasons whose `direction` field matches the input, and whose text
                            wording agrees with it ("argues against" present iff argues_against)
  uncertainty (C5)          uncertain -> the note says so and asks for clinician review; surrogate disagreement
                            and unreliable segmentation are stated when present
  invented_terms            dermatology vocabulary in the text that the input does not contain
  digits                    any digit (the input has none, so a digit is an unsupported claim)
  named_diagnosis           a specific diagnosis named (the model is binary)
  advice                    treatment/management advice
  confident_terms (C6)      overconfident wording ("clearly", "definitely", ...), counted for the C6 comparison
"""
import json
import re

# Dermatology vocabulary the explanations could plausibly invent. A term is "invented" when it appears in the
# explanation but nowhere in the input's concept/measure/finding/relative_to_benign/limitations text.
# Whole-word match; a trailing * marks a stem ("elevat*" matches elevated, elevation).
CONCEPT_VOCAB = [
    "pigment network", "network", "globule*", "dot*", "streak*", "pseudopod*", "veil", "regression", "vessel*",
    "vascular*", "telangiect*", "arboriz*", "ulcer*", "crust*", "scal*", "keratin*", "milia", "comedo*",
    "fissur*", "blue-white", "blue-gray", "blue", "black", "white", "red", "pink", "brown", "yellow", "gray",
    "grey", "diameter", "millimet*", "mm", "large", "small", "size", "evolution", "evolv*", "growth", "growing",
    "changing", "itch*", "bleed*", "elevat*", "raised", "nodul*", "papul*", "macul*", "plaque*", "border*",
    "asymmetr*", "symmetr*", "colo*r*", "textur*", "rough*", "smooth*", "shape*", "dark*", "light*", "pigment*",
    "erythem*",
]


def _pattern(term):
    stem = term.endswith("*")
    body = re.escape(term.rstrip("*")).replace(r"\*", r"\w*")   # inner * (colo*r*) = optional letters
    return re.compile(r"\b" + body + (r"\w*" if stem else r"\b"), re.I)


VOCAB_PATTERNS = {t: _pattern(t) for t in CONCEPT_VOCAB}
DIAGNOSES = ["melanoma", "carcinoma", "bcc", "scc", "nevus", "naevus", "keratosis", "lentigo", "dermatofibroma",
             "angioma", "actinic", "bowen", "keratoacanthoma", "mole"]
ADVICE = ["excis", "biops", "remove", "treat", "cream", "surgery", "follow-up in", "refer to", "prescri",
          "should be excised", "monitor"]
CONFIDENT = ["clearly", "definitely", "certainly", "undoubtedly", "confirms", "confirm", "conclusive",
             "without doubt", "obvious", "highly suggestive", "strongly suggests", "classic", "typical of malignan",
             "characteristic of"]
UNCERTAIN_WORDS = ["uncertain", "uncertainty", "not sure", "low confidence"]
REVIEW_WORDS = ["review", "clinician", "dermatologist", "expert"]
SURROGATE_WORDS = ["may not reflect", "does not reproduce", "not reproduce", "approximat", "may not explain",
                   "disagree"]
SEGMENT_WORDS = ["outline", "segmentation", "border detection", "boundary"]
AGAINST_WORDS = ["argues against", "argue against", "against the", "does not support", "counts against",
                 "speaks against", "weighs against"]


def parse(raw):
    """Model output (str) or dict -> (dict | None, parse_ok). Takes the first {...} block in a string."""
    if isinstance(raw, dict):
        return raw, True
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None, False
    try:
        return json.loads(match.group(0)), True
    except json.JSONDecodeError:
        return None, False


def _text(out):
    parts = [out.get("summary", ""), out.get("uncertainty_note", "")]
    parts += [r.get("text", "") for r in out.get("reasons", []) if isinstance(r, dict)]
    return " ".join(p for p in parts if isinstance(p, str))


def _has(text, words):
    t = text.lower()
    return [w for w in words if w in t]


def verify(llm_input, raw):
    out, parse_ok = parse(raw)
    res = {"parse_ok": parse_ok}
    required = {"summary", "reasons", "uncertainty_note", "limitations"}
    res["schema_ok"] = bool(out) and required <= set(out) and isinstance(out.get("reasons"), list)
    if not res["schema_ok"]:
        return res | {"pass_all": False}

    evidence = {e["feature_id"]: e for e in llm_input["evidence"]}
    reasons = [r for r in out["reasons"] if isinstance(r, dict)]
    grounded = [r for r in reasons if r.get("feature_id") in evidence]
    res["n_reasons"] = len(reasons)
    res["grounding"] = len(grounded) / len(reasons) if reasons else 0.0
    res["coverage"] = len({r["feature_id"] for r in grounded}) / len(evidence)

    dir_ok = []
    for r in grounded:
        expected = evidence[r["feature_id"]]["direction"]
        says_against = bool(_has(r.get("text", ""), AGAINST_WORDS))
        dir_ok.append(r.get("direction") == expected and says_against == (expected == "argues_against"))
    res["direction"] = sum(dir_ok) / len(dir_ok) if dir_ok else 0.0

    note = out.get("uncertainty_note", "") or ""
    full = _text(out)
    checks = []
    if llm_input["uncertain"]:
        checks.append(bool(_has(note, UNCERTAIN_WORDS)) and bool(_has(note, REVIEW_WORDS)))
    if not llm_input["surrogate_agrees"]:
        checks.append(bool(_has(full, SURROGATE_WORDS)))
    if not llm_input["segmentation_reliable"]:
        checks.append(bool(_has(full, SEGMENT_WORDS)))
    res["uncertainty_ok"] = all(checks)
    res["uncertainty_flag_ok"] = bool(_has(note, UNCERTAIN_WORDS)) if llm_input["uncertain"] else True

    allowed = " ".join([e["measure"] + " " + e["finding"] + " " + e["relative_to_benign"] + " " + e["concept"]
                        for e in llm_input["evidence"]] + llm_input["limitations"]).lower()
    res["invented_terms"] = sorted(t for t, pat in VOCAB_PATTERNS.items() if pat.search(full) and not pat.search(allowed))
    res["digits"] = bool(re.search(r"\d", full))
    res["named_diagnosis"] = sorted({w for w in DIAGNOSES if re.search(rf"\b{w}", full.lower())})
    res["advice"] = _has(full, ADVICE)
    res["confident_terms"] = _has(full, CONFIDENT)
    res["pass_all"] = (res["grounding"] == 1.0 and res["coverage"] == 1.0 and res["direction"] == 1.0
                       and res["uncertainty_ok"] and not res["invented_terms"] and not res["digits"]
                       and not res["named_diagnosis"] and not res["advice"])
    return res


def summarize(results):
    """Aggregate per-explanation results into the C3/C4/C5 rates."""
    import numpy as np
    n = len(results)
    ok = [r for r in results if r.get("schema_ok")]
    mean = lambda k: float(np.mean([r[k] for r in ok])) if ok else 0.0  # noqa: E731
    return {
        "n": n,
        "parse_rate": sum(r["parse_ok"] for r in results) / n if n else 0.0,
        "schema_rate": len(ok) / n if n else 0.0,
        "grounding_C3": mean("grounding"),
        "coverage": mean("coverage"),
        "direction_C4": mean("direction"),
        "uncertainty_C5": mean("uncertainty_ok"),
        "invented_term_rate": float(np.mean([bool(r["invented_terms"]) for r in ok])) if ok else 0.0,
        "digit_rate": float(np.mean([r["digits"] for r in ok])) if ok else 0.0,
        "named_diagnosis_rate": float(np.mean([bool(r["named_diagnosis"]) for r in ok])) if ok else 0.0,
        "advice_rate": float(np.mean([bool(r["advice"]) for r in ok])) if ok else 0.0,
        "confident_term_rate": float(np.mean([bool(r["confident_terms"]) for r in ok])) if ok else 0.0,
        "pass_all_rate": float(np.mean([r["pass_all"] for r in results])) if n else 0.0,
    }
