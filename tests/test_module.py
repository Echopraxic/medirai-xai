"""Explanation module (explain/module.py, T8.1): one call per case, verifier-gated LLM text, audiences, French."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, STUBBED

pytestmark = pytest.mark.skipif("shap" in STUBBED, reason="needs shap")

sys.path.insert(0, str(ROOT / "explain"))
sys.path.insert(0, str(ROOT / "trees"))
sys.path.insert(0, str(ROOT / "concepts"))

FEATS = ["A_shape_asym_major", "B_radial_cv", "C_darkness_vs_skin", "T_glcm_contrast", "S_hu3"]


@pytest.fixture(scope="module")
def explainer():
    import xgboost as xgb
    from module import Explainer
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATS))), columns=FEATS)
    y = ((X.B_radial_cv + X.C_darkness_vs_skin + 0.3 * rng.normal(size=400)) > 0).astype(int)
    models = [xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=s).fit(X, y) for s in (0, 1)]
    ref = {f: np.sort(X.loc[y == 0, f].to_numpy()) for f in FEATS}
    return Explainer(models[0], models[1], FEATS, ref, (0.1, 0.5), entropy_threshold=0.5)


CASE = {"A_shape_asym_major": 0.2, "B_radial_cv": 2.0, "C_darkness_vs_skin": 1.5, "T_glcm_contrast": 0.0, "S_hu3": -0.1}


def test_template_explanation_is_grounded_and_verified(explainer):
    out = explainer.explain(CASE, {"p_malignant": 0.9, "entropy_of_expected": 0.1}, "ok")
    assert out["verification"]["pass_all"] and out["fallback_used"] is None
    assert [r["feature_id"] for r in out["structured"]["reasons"]] == [c["feature_id"] for c in out["top_concepts"]]
    assert out["structured"]["reasons"][0]["feature_id"] in ("B_radial_cv", "C_darkness_vs_skin")
    assert out["glossary"] == [] and "What these terms mean" not in out["text"]
    assert out["provenance"]["input_hash"] and out["provenance"]["audience"] == "dermatologist"


def test_uncertain_case_says_so_first(explainer):
    out = explainer.explain(CASE, {"p_malignant": 0.55, "entropy_of_expected": 0.9}, "fail")
    note = out["structured"]["uncertainty_note"]
    assert "uncertain" in note and "outline" in note
    assert out["text"].index("uncertain") < out["text"].index("Reasons:")


class BadLLM:
    """Backend that invents a structure and a number: the module must not return its text."""
    name, revision = "bad-llm", "test"

    def generate(self, batch_msgs, batch_inputs):
        return [json.dumps({"reasons": [{"feature_id": e["feature_id"], "direction": e["direction"],
                                         "text": "Blue-white veil covers 40% of the lesion."}
                                        for e in inp["evidence"]]}) for inp in batch_inputs]


def test_failed_llm_output_falls_back_to_template(explainer):
    import module
    llm = module.Explainer(explainer.surrogate, explainer.gt_model, FEATS, explainer.ref, explainer.thresholds,
                           0.5, backend=BadLLM(), prompt="hybrid_v1")
    out = llm.explain(CASE, {"p_malignant": 0.9, "entropy_of_expected": 0.1}, "ok")
    assert set(out["fallback_used"]) >= {"invented_terms", "digits"}
    assert out["provenance"]["backend"] == "template (fallback)" and out["verification"]["pass_all"]
    assert "veil" not in out["text"] and "40" not in out["text"]


def test_gp_glossary_covers_cited_concepts_and_avoids_banned_terms(explainer):
    from avoided_terms import hits
    out = explainer.explain(CASE, {"p_malignant": 0.9, "entropy_of_expected": 0.1}, "ok", audience="general_practitioner")
    cited = {c["concept"] for c in out["top_concepts"]}
    assert len(out["glossary"]) == len(cited) and "What these terms mean:" in out["text"]
    assert out["structured"]["reasons"] == explainer.explain(CASE, {"p_malignant": 0.9, "entropy_of_expected": 0.1},
                                                             "ok")["structured"]["reasons"]   # same reasons for both
    import module
    for en, fr in module.GLOSSARY.values():
        assert not hits(en, "en") and not hits(fr, "fr")


def test_french_uses_template_layer_and_quebec_colon(explainer):
    out = explainer.explain(CASE, {"p_malignant": 0.1, "entropy_of_expected": 0.1}, "ok", language="fr")
    assert out["provenance"]["backend"] == "template_fr" and out["text"].startswith("Le modèle prédit")
    assert "Raisons :" in out["text"]
    with pytest.raises(ValueError):
        explainer.explain(CASE, {"p_malignant": 0.1, "entropy_of_expected": 0.1}, "ok", audience="patient")
