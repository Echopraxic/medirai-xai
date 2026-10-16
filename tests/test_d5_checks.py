"""D5 automated checks (plan T6.1-T6.3): each check must catch a deliberately unfaithful output."""
import copy
import json
import sys

import numpy as np

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
import consistency  # noqa: E402
import counterfactuals as cf  # noqa: E402
import llm_input as li  # noqa: E402
import nli_check  # noqa: E402
from template import render_llm_input  # noqa: E402


def payload(directions=("supports", "supports", "argues_against"), ids=("B_radial_cv", "C_darkness_vs_skin", "T_roughness")):
    groups = {"B": "border", "C": "color", "T": "texture/surface", "A": "asymmetry"}
    return {"prediction": "malignant", "probability_malignant": 0.9, "uncertainty": {"entropy_of_expected": 0.1},
            "surrogate_agrees": True, "audience": "dermatologist",
            "top_concepts": [{"concept": groups[f[0]], "group": f[0], "feature_id": f, "direction": d,
                              "strength": "strong", "grouped_shap": 1.0 if d == "supports" else -1.0,
                              "feature_value": 0.5, "benign_percentile": 90.0} for f, d in zip(ids, directions)]}


def gen(pl, variant="base", output=None):
    inp = li.build(pl, "ok", entropy_threshold=0.5)
    out = output if output is not None else render_llm_input(inp)
    return {"isic_id": "ISIC_X", "row_id": f"ISIC_X#{variant}", "variant": variant, "stratum": "correct|confident",
            "llm_input": inp, "final_output": out, "raw_output": json.dumps(out)}


def test_counterfactual_flip_is_caught_when_the_output_ignores_it():
    base = gen(payload())
    flipped = payload(directions=("argues_against", "supports", "argues_against"))
    assert cf.score_case(base, gen(flipped, "flip"))["sensitive"]
    stale = gen(flipped, "flip", output=copy.deepcopy(base["final_output"]))  # output did not follow the flip
    assert not cf.score_case(base, stale)["sensitive"]


def test_counterfactual_drop_and_swap():
    base = gen(payload())
    dropped = payload(directions=("supports", "supports"), ids=("B_radial_cv", "C_darkness_vs_skin"))
    assert cf.score_case(base, gen(dropped, "drop"))["sensitive"]
    assert not cf.score_case(base, gen(dropped, "drop", output=copy.deepcopy(base["final_output"])))["sensitive"]
    swapped = payload(ids=("B_radial_cv", "C_darkness_vs_skin", "A_shape_asym_max"))
    res = cf.score_case(base, gen(swapped, "swap"))
    assert res["sensitive"] and all(res["stable_ok"])


def test_repeat_determinism():
    base = gen(payload())
    assert cf.score_case(base, gen(payload(), "repeat"))["deterministic"]
    other = gen(payload(), "repeat")
    other["raw_output"] += " "
    assert not cf.score_case(base, other)["deterministic"]


class TokenStub:
    """Entails iff the hypothesis has the same word set as the premise; otherwise contradiction."""
    name = "stub"

    def __call__(self, pairs):
        tok = cf.tokens
        return np.array([[1, 0, 0] if tok(p) == tok(h) else [0, 0, 1] for p, h in pairs], float)


def test_nli_claims_and_calibration_wiring(tmp_path):
    g = gen(payload())
    pairs = nli_check.claim_pairs(g)
    assert len(pairs) == 3 and all(f in {"B_radial_cv", "C_darkness_vs_skin", "T_roughness"} for _, _, f in pairs)
    rows = [{"isic_id": f"I{i}", "split": s, "label": 1, "mask_quality": "ok", "source": "MILK", "llm_input": payload()}
            for i, s in enumerate(["val"] * 3 + ["test"] * 5)]
    path = tmp_path / "inputs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rep = nli_check.calibrate(path, TokenStub(), limit=5)
    assert rep["false_alarm_rate"] == 0 and all(v == 1 for v in rep["detection_rate"].values())


def test_similar_inputs_get_similar_claims(tmp_path):
    gens = [gen(payload()) for _ in range(3)]
    path = tmp_path / "g.jsonl"
    path.write_text("\n".join(json.dumps(g) for g in gens), encoding="utf-8")
    rep = consistency.similar(path)
    assert rep["same_signature"]["identical_rate"] == 1.0
