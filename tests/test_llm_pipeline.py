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
    assert border["finding"] == "irregular" and border["relative_to_benign"] == "much more irregular than most benign lesions"
    assert color["finding"] == "dark" and color["relative_to_benign"] == "darker than most benign lesions"
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


def test_supporting_the_other_class_counts_as_arguing_against(inp):
    """Smoke run 2026-10-07: 'supporting the benign prediction' for a malignant prediction is a correct paraphrase."""
    out = good(inp)
    out["reasons"][2]["text"] = "The shape pattern is unusual; this strongly supports the benign prediction."
    assert verify(inp, out)["direction"] == 1.0
    out["reasons"][0]["text"] = "The border outline is irregular; this strongly supports the benign prediction."
    assert verify(inp, out)["direction"] < 1.0   # input says this one supports malignant


def test_hybrid_assembles_code_caveats_around_llm_reasons():
    """hybrid_v1: caveats come from code, so they hold even when the LLM writes only reasons."""
    from llm_generate import assemble, load_prompt
    spec = load_prompt("hybrid_v1")
    inp = li.build(raw_payload(entropy=0.9, agrees=False), "review", 0.5)
    raw = json.dumps({"reasons": render_llm_input(inp)["reasons"]})
    final = assemble(spec, inp, raw)
    res = verify(inp, final)
    assert res["pass_all"] and res["uncertainty_ok"]
    assert "outline" in final["uncertainty_note"] and "mixed" in final["summary"]
    assert not verify(inp, assemble(spec, inp, "no json here"))["pass_all"]   # unusable output fails, never passes


def test_independent_check_caveat_only_for_confident_predictions_with_feature_model_doubt():
    doubtful = dict(raw_payload(), feature_model_p_predicted=0.40)     # doubt 0.60 > cut
    supported = dict(raw_payload(), feature_model_p_predicted=0.90)
    assert li.build(doubtful, "ok", entropy_threshold=0.5)["independent_check_weak"]
    assert not li.build(supported, "ok", entropy_threshold=0.5)["independent_check_weak"]
    # uncertain cases already carry the stronger uncertainty note; pre-v2 inputs have no field
    assert not li.build(dict(doubtful, uncertainty={"entropy_of_expected": 0.9}), "ok", 0.5)["independent_check_weak"]
    assert not li.build(raw_payload(), "ok", 0.5)["independent_check_weak"]


def test_verifier_requires_the_independent_check_caveat():
    inp = li.build(dict(raw_payload(), feature_model_p_predicted=0.40), "ok", entropy_threshold=0.5)
    out = render_llm_input(inp)
    assert "independent model" in out["uncertainty_note"] and verify(inp, json.dumps(out))["uncertainty_ok"]
    out["uncertainty_note"] = ""
    assert not verify(inp, json.dumps(out))["uncertainty_ok"]


def test_benign_comparison_follows_the_finding_word():
    # asymmetry score below most benign lesions -> "symmetric", and the comparison must agree with that word
    assert li.relative(15.0, "symmetric") == "more symmetric than most benign lesions"
    assert li.relative(5.0, "symmetric") == "much more symmetric than most benign lesions"
    assert li.relative(95.0, "dark") == "much darker than most benign lesions"
    assert li.relative(80.0, "high") == "higher than in most benign lesions"
    assert li.relative(50.0, "typical") == "within the typical range of benign lesions"


def test_deferral_flag_overrides_entropy_threshold():
    """S1-4: an attached uncertainty-table flag decides 'uncertain'; without it the entropy threshold does."""
    import sys
    sys.path.insert(0, str(ROOT / "explain"))
    import attach_deferral
    import llm_input as li
    import pandas as pd
    rows = [{"isic_id": "A", "split": "test", "llm_input": {"uncertainty": {"entropy_of_expected": 0.01}}},
            {"isic_id": "B", "split": "test", "llm_input": {"uncertainty": {"entropy_of_expected": 0.9}}}]
    table = pd.DataFrame({"isic_id": ["A", "B"], "deferred": [True, False], "defer_reason": ["disagreement", ""]})
    assert not li.is_uncertain(rows[0]["llm_input"], 0.5) and li.is_uncertain(rows[1]["llm_input"], 0.5)
    out = attach_deferral.attach(rows, table)
    assert li.is_uncertain(out[0]["llm_input"], 0.5) and not li.is_uncertain(out[1]["llm_input"], 0.5)
