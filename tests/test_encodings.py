"""Attribution encodings for the paper's RQ2 and the verifier's number rule."""
import json
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
import attribution_encodings as enc  # noqa: E402
from template import render_llm_input  # noqa: E402
from verifier import verify  # noqa: E402


def payload():
    return {"prediction": "malignant", "probability_malignant": 0.9, "uncertainty": {"entropy_of_expected": 0.1},
            "surrogate_agrees": True, "audience": "dermatologist",
            "top_concepts": [{"concept": "border", "group": "B", "feature_id": "B_radial_cv", "direction": "supports",
                              "strength": "strong", "grouped_shap": 1.25, "feature_value": 0.31, "benign_percentile": 95.0},
                             {"concept": "color", "group": "C", "feature_id": "C_darkness_vs_skin",
                              "direction": "argues_against", "strength": "weak", "grouped_shap": -0.4,
                              "feature_value": 12.5, "benign_percentile": 60.0}],
            "top_features": [{"feature_id": "B_radial_cv", "group": "B", "shap": 0.9, "feature_value": 0.31,
                              "benign_percentile": 95.0},
                             {"feature_id": "T_roughness", "group": "T", "shap": -0.2, "feature_value": 0.05,
                              "benign_percentile": 20.0}]}


def test_each_encoding_shows_only_its_format():
    shown = {e: enc.encode(payload(), "ok", 0.5, e)[0]["evidence"] for e in enc.ENCODINGS}
    assert shown["C-raw"][0] == {"feature_id": "B_radial_cv", "concept": "border", "shap_toward_prediction": 1.25}
    assert set(shown["C-bin"][0]) == {"feature_id", "concept", "direction", "strength"}
    assert [e["feature_id"] for e in shown["F-raw"]] == ["B_radial_cv", "T_roughness"]  # ungrouped features
    assert "measure" in shown["C-phrase"][0] and not any(ch.isdigit() for ch in json.dumps(shown["C-phrase"]))


def test_f_raw_verify_input_matches_its_cited_features():
    _, verify_input, allowed = enc.encode(payload(), "ok", 0.5, "F-raw")
    assert [e["feature_id"] for e in verify_input["evidence"]] == ["B_radial_cv", "T_roughness"]
    assert [e["direction"] for e in verify_input["evidence"]] == ["supports", "argues_against"]
    assert "0.9" in allowed and "0.31" in allowed


def test_verifier_allows_copied_numbers_but_not_invented_ones():
    _, inp, allowed = enc.encode(payload(), "ok", 0.5, "C-raw")
    out = render_llm_input(inp)
    out["reasons"][0]["text"] += " Its contribution is 1.250."           # copied (same number)
    assert not verify(inp, json.dumps(out), allowed)["digits"]
    out["reasons"][0]["text"] += " About 7 mm across."                    # invented
    assert verify(inp, json.dumps(out), allowed)["digits"]


def test_f_raw_refuses_old_inputs_without_raw_features():
    old = payload()
    del old["top_features"]
    with pytest.raises(ValueError):
        enc.encode(old, "ok", 0.5, "F-raw")
