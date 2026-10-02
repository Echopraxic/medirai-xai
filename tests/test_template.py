"""No-LLM template baseline: rule-checked guarantees (C4 direction wording, C5 uncertainty flag)."""
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
from template import PHRASES, render  # noqa: E402


def payload(entropy=0.1, agrees=True, directions=("supports", "supports", "supports")):
    reasons = [{"concept": c, "group": g, "feature_id": f, "direction": d, "strength": "strong",
                "grouped_shap": 1.0, "feature_value": 0.5, "benign_percentile": 90.0}
               for (c, g, f), d in zip([("border", "B", "B_radial_cv"), ("color", "C", "C_color_entropy"),
                                        ("asymmetry", "A", "A_shape_asym_max")], directions)]
    return {"prediction": "malignant", "probability_malignant": 0.81, "uncertainty": {"entropy_of_expected": entropy},
            "surrogate_agrees": agrees, "top_concepts": reasons, "audience": "dermatologist"}


def test_confident_case_has_no_hedges_and_cites_every_feature():
    exp = render(payload(), entropy_threshold=0.5)
    assert exp.hedges == [] and exp.uncertainty_note == ""
    assert [r["feature_id"] for r in exp.reasons] == ["B_radial_cv", "C_color_entropy", "A_shape_asym_max"]
    assert all("supports the malignant prediction" in r["text"] for r in exp.reasons)


def test_uncertain_case_is_always_flagged():
    exp = render(payload(entropy=0.6), entropy_threshold=0.5)
    assert "uncertain" in exp.hedges and "clinician" in exp.uncertainty_note


def test_surrogate_disagreement_and_mixed_evidence_are_stated():
    exp = render(payload(agrees=False, directions=("supports", "argues_against", "supports")), entropy_threshold=0.5)
    assert set(exp.hedges) == {"surrogate_disagrees", "mixed_evidence"}
    assert "argues against the malignant prediction" in exp.reasons[1]["text"]


def test_every_feature_has_a_phrase():
    sys.path.insert(0, str(ROOT / "features"))
    import extract
    assert set(extract.FEATURE_DICTIONARY) <= set(PHRASES)
