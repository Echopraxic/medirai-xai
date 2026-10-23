"""Quebec French explanation layer (explain/lang_fr.py): coverage, same facts as English, typography."""
import re
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
import lang_fr as F  # noqa: E402
from template import PHRASES, render_llm_input  # noqa: E402


def _inp(fid="A_shape_asym_max", finding="asymmetric", rel="much more asymmetric than most benign lesions",
         direction="supports", uncertain=True):
    return {"prediction": "malignant", "confidence": "low", "uncertain": uncertain, "surrogate_agrees": False,
            "independent_check_weak": False, "segmentation_reliable": False,
            "limitations": ["x"],
            "evidence": [{"feature_id": fid, "concept": "asymmetry", "measure": PHRASES[fid][0], "finding": finding,
                          "relative_to_benign": rel, "direction": direction, "strength": "weak",
                          "clinically_mapped": True}]}


def test_every_phrased_feature_has_french():
    assert set(PHRASES) - set(F.PHRASES_FR) == {f"S_hu{i}" for i in range(1, 8)}


def test_french_states_the_same_facts():
    fr = F.render_llm_input_fr(_inp())
    en = render_llm_input(_inp())
    assert len(fr["reasons"]) == len(en["reasons"])
    text = fr["reasons"][0]["text"]
    assert "asymétrique" in text and "nettement plus asymétrique" in text and "appuie faiblement" in text
    # every caveat the English version emits has its French twin
    assert F.NOTES["uncertain"] in fr["uncertainty_note"] and F.NOTES["surrogate"] in fr["uncertainty_note"]
    assert F.NOTES["outline"] in fr["uncertainty_note"]
    against = F.render_llm_input_fr(_inp(direction="argues_against"))
    assert "va faiblement à l'encontre" in against["reasons"][0]["text"] and F.NOTES["mixed"] in against["summary"]


def test_quantity_and_typical_bands():
    hi = F.render_llm_input_fr(_inp("C_frac_red", "high", "much higher than in most benign lesions"))
    assert "nettement au-dessus de la plupart des lésions bénignes" in hi["reasons"][0]["text"]
    typ = F.render_llm_input_fr(_inp("C_frac_red", "typical", "within the typical range of benign lesions"))
    assert "typique (dans l'intervalle habituel des lésions bénignes)" in typ["reasons"][0]["text"]


def test_quebec_typography_and_no_anglicism():
    text = F.render_llm_input_fr(_inp())["reasons"][0]["text"]
    assert f"{F.NBSP}:" in text and " ;" not in text and "« maligne »" in text
    assert not re.search(r"\bsupporte", text)


def test_avoided_terms_never_flag_our_own_wording():
    sys.path.insert(0, str(ROOT / "concepts"))
    import avoided_terms as A
    from llm_input import LIMITATIONS
    en = [" ".join(v) for v in PHRASES.values()] + [" ".join(LIMITATIONS)]
    fr = [" ".join(v) for v in F.PHRASES_FR.values()] + [" ".join(F.LIMITATIONS_FR), F.UNMAPPED_SHAPE] + list(F.NOTES.values())
    assert not [t for t in en if A.hits(t, "en")]
    assert not [t for t in fr if A.hits(t, "fr")]
    # a full rendered explanation is clean in both languages, while planted terms are caught
    for render, lang in ((render_llm_input, "en"), (F.render_llm_input_fr, "fr")):
        out = render(dict(_inp(), limitations=LIMITATIONS))
        text = " ".join([out["summary"], out["uncertainty_note"], out["limitations"]] + [r["text"] for r in out["reasons"]])
        assert not A.hits(text, lang), A.hits(text, lang)
    assert A.hits("Un érythème et une croûte sont visibles; il s'agit clairement d'un mélanome.", "fr") == \
        ["diagnostics nommés", "formulations de certitude", "ulcération / croûte", "érythème"]
    assert A.hits("Large lesion with scale; biopsy recommended.", "en") == \
        ["diameter / size in mm / large / small", "management advice", "scale / scaling"]
