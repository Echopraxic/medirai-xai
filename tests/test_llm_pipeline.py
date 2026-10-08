"""LLM input builder and verifier (plan T4.2, T6.1): the input is number-free and each verifier rule fires."""
import copy
import json
import re
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
import llm_input as li  # noqa: E402
from template import render_llm_input  # noqa: E402
from verifier import verify  # noqa: E402


def raw_payload(entropy=0.1, agrees=True, directions=("supports", "supports", "argues_against")):
    feats = [("border", "B", "B_radial_cv", 95.0), ("color", "C", "C_darkness_vs_skin", 80.0),
             ("shape", "S", "S_hu3", 5.0)]
    return {"prediction": "malignant", "probability_malignant": 0.91,
            "uncertainty": {"entropy_of_expected": entropy}, "surrogate_agrees": agrees, "audience": "dermatologist",
            "top_concepts": [{"concept": c, "group": g, "feature_id": f, "direction": d, "strength": "strong",
                              "grouped_shap": 1.23, "feature_value": 0.456, "benign_percentile": p}
                             for (c, g, f, p), d in zip(feats, directions)]}


@pytest.fixture
def inp():
    return li.build(raw_payload(), "ok", entropy_threshold=0.5)


def test_llm_input_has_no_numbers_and_categorical_findings(inp):
    shown = {k: v for k, v in inp.items() if k not in ("schema_version", "phrase_bank", "evidence")}
    shown["evidence"] = [{k: v for k, v in e.items() if k != "feature_id"} for e in inp["evidence"]]
    assert not re.search(r"\d", json.dumps(shown))   # feature ids (S_hu3) are copied, never shown as numbers
    border, color, hu = inp["evidence"]
    assert border["finding"] == "irregular" and border["relative_to_benign"].startswith("much higher")
    assert color["finding"] == "dark" and color["relative_to_benign"].startswith("higher")
    assert not hu["clinically_mapped"] and "without a clinical name" in hu["measure"]
    assert inp["confidence"] == "high" and inp["uncertain"] is False


def test_template_on_llm_input_passes_every_rule(inp):
    for case in (inp, li.build(raw_payload(entropy=0.9, agrees=False), "review", 0.5)):
        assert verify(case, render_llm_input(case))["pass_all"]


def good(inp):
    return render_llm_input(inp)


@pytest.mark.parametrize("mutate, failed", [
    (lambda o: o["reasons"].__setitem__(0, {**o["reasons"][0], "feature_id": "B_fractal_dim"}), "grounding"),
    (lambda o: o["reasons"].pop(), "coverage"),
    (lambda o: o["reasons"][2].__setitem__("text", "The shape pattern supports the prediction."), "direction"),
    (lambda o: o["reasons"][0].__setitem__("text", o["reasons"][0]["text"] + " Blue-white veil is present."),
     "invented_terms"),
    (lambda o: o.__setitem__("summary", "Malignant with 91% probability."), "digits"),
    (lambda o: o.__setitem__("summary", "This looks like a melanoma."), "named_diagnosis"),
    (lambda o: o.__setitem__("uncertainty_note", "Excision is recommended."), "advice"),
])
def test_verifier_catches_each_failure(inp, mutate, failed):
    out = copy.deepcopy(good(inp))
    mutate(out)
    res = verify(inp, out)
    assert not res["pass_all"]
    if failed in ("grounding", "coverage", "direction"):
        assert res[failed] < 1.0, res
    else:
        assert res[failed], res


def test_uncertain_case_without_flag_fails():
    inp = li.build(raw_payload(entropy=0.9), "ok", 0.5)
    out = good(inp)
    out["uncertainty_note"] = ""
    res = verify(inp, out)
    assert not res["uncertainty_ok"] and not res["pass_all"]


def test_unparseable_output_fails_cleanly(inp):
    res = verify(inp, "Sure! Here is the explanation: the lesion is irregular.")
    assert res == {"parse_ok": False, "schema_ok": False, "pass_all": False}
    assert verify(inp, "```json\n" + json.dumps(good(inp)) + "\n```")["pass_all"]
