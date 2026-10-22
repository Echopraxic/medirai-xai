"""Explanation features after the 2026-10-21 review: encoding labels (not filtering), scope flag + verifier rule,
concept-share sentence, LLM-path labels added in code."""
import copy
import json
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
sys.path.insert(0, str(ROOT / "trees"))
import llm_input as li  # noqa: E402
from lang_fr import render_llm_input_fr  # noqa: E402
from llm_generate import assemble  # noqa: E402
from template import render_llm_input  # noqa: E402
from verifier import verify  # noqa: E402

from test_llm_pipeline import raw_payload  # noqa: E402


def payload_with(**extra):
    p = raw_payload()
    p["top_concepts"][0]["model_encoded"] = False     # B_radial_cv: outline feature the ResNet does not encode
    p["top_concepts"][1]["model_encoded"] = True
    return {**p, **extra}


def test_unencoded_reason_is_labelled_not_dropped_in_en_and_fr():
    inp = li.build(payload_with(), "ok", 0.5)
    assert [e["feature_id"] for e in inp["evidence"]] == ["B_radial_cv", "C_darkness_vs_skin", "S_hu3"]
    out = render_llm_input(inp)
    assert "outline measure the image model does not appear to use" in out["reasons"][0]["text"]
    assert "does not appear" not in out["reasons"][1]["text"]
    assert verify(inp, out)["pass_all"]
    assert "ne semble pas utiliser" in render_llm_input_fr(inp)["reasons"][0]["text"]


def test_old_inputs_keep_their_hash():
    old = li.build(raw_payload(), "ok", 0.5)
    assert "in_scope" not in old and "named_concept_share" not in old
    assert all("model_encoded" not in e for e in old["evidence"])


def test_llm_reasons_get_the_label_in_code():
    inp = li.build(payload_with(), "ok", 0.5)
    raw = json.dumps({"reasons": [{"feature_id": e["feature_id"], "direction": e["direction"],
                                   "text": f"The {e['measure']} {e['direction'].replace('_', ' ')} the prediction."}
                                  for e in inp["evidence"]]})
    final = assemble({"mode": "reasons_only"}, inp, raw)
    assert final["reasons"][0]["text"].endswith("(an outline measure the image model does not appear to use).")
    assert "does not appear" not in final["reasons"][1]["text"]


def test_out_of_scope_needs_a_scope_warning():
    inp = li.build(payload_with(in_scope=False), "ok", 0.5)
    good = render_llm_input(inp)
    assert "differs from the images the model was trained on" in good["uncertainty_note"]
    assert verify(inp, good)["pass_all"]
    bad = copy.deepcopy(good)
    bad["uncertainty_note"] = ""
    assert not verify(inp, bad)["uncertainty_ok"]
    assert "Cette image diffère" in render_llm_input_fr(inp)["uncertainty_note"]


def test_concept_share_sentence_only_when_set():
    assert "named features" not in render_llm_input(li.build(payload_with(), "ok", 0.5))["summary"]
    inp = li.build(payload_with(named_concept_share="little"), "ok", 0.5)
    out = render_llm_input(inp)
    assert "account for little of this decision" in out["summary"] and verify(inp, out)["pass_all"]


def test_case_reasons_label_does_not_change_the_choice():
    import numpy as np
    import pandas as pd
    from treeshap_top3 import case_reasons
    feats = ["A_shape_asym_max", "B_radial_cv", "C_darkness_vs_skin", "T_roughness"]
    signed = pd.Series([0.1, 0.9, -0.5, 0.3], index=feats)
    groups = pd.Series({"A": 0.1, "B": 0.9, "C": -0.5, "T": 0.3})
    values = pd.Series([1.0, 2.0, 3.0, 4.0], index=feats)
    ref = {f: np.arange(10.0) for f in feats}
    plain = case_reasons(groups, signed, values, feats, ref, (0.2, 0.6))
    lab = case_reasons(groups, signed, values, feats, ref, (0.2, 0.6), not_encoded={"B_radial_cv"})
    assert [r["feature_id"] for r in plain] == [r["feature_id"] for r in lab] == ["B_radial_cv", "C_darkness_vs_skin", "T_roughness"]
    assert "model_encoded" not in plain[0] and lab[0]["model_encoded"] is False and lab[1]["model_encoded"] is True
    filt = case_reasons(groups, signed, values, feats, ref, (0.2, 0.6), not_cited={"B_radial_cv"})
    assert [r["feature_id"] for r in filt] == ["C_darkness_vs_skin", "T_roughness", "A_shape_asym_max"]


def test_darker_skin_flag_needs_a_warning_and_review_in_en_and_fr():
    inp = li.build(payload_with(tone_unvalidated=True), "ok", 0.5)
    assert inp["tone_unvalidated"] is True
    good = render_llm_input(inp)
    assert "darker than the skin the model was mainly validated on" in good["uncertainty_note"]
    assert verify(inp, good)["pass_all"]
    bad = copy.deepcopy(good)
    bad["uncertainty_note"] = "The model is uncertain and should be reviewed by a clinician."
    inp_uncertain = li.build(payload_with(tone_unvalidated=True, uncertainty={"entropy_of_expected": 0.9}), "ok", 0.5)
    assert not verify(inp_uncertain, bad)["uncertainty_ok"]          # uncertainty wording alone does not cover the tone flag
    assert "peaux plus foncées" in render_llm_input_fr(inp)["uncertainty_note"]
    assert "tone_unvalidated" not in li.build(payload_with(), "ok", 0.5)   # absent unless flagged: input hashes unchanged
