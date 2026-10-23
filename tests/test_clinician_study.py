"""Clinician study builder (eval/clinician_study/build_survey.py): stratification, counterbalancing, blinding,
bilingual LimeSurvey structure."""
import sys
from collections import Counter

from conftest import ROOT

sys.path.insert(0, str(ROOT / "eval" / "clinician_study"))
import build_survey as B  # noqa: E402


def _inp(pred="malignant", uncertain=False):
    return {"prediction": pred, "confidence": "moderate", "uncertain": uncertain, "surrogate_agrees": True,
            "independent_check_weak": False, "segmentation_reliable": True, "limitations": ["Limitation."],
            "evidence": [{"feature_id": "C_frac_red", "concept": "color", "measure": "share of red-toned areas",
                          "finding": "high", "relative_to_benign": "higher than in most benign lesions",
                          "direction": "supports", "strength": "moderate", "clinically_mapped": True}]}


def _pool(n_per=7):
    pool = []
    for label in (0, 1):
        for correct in (True, False):
            for unc in (True, False):
                for j in range(n_per):
                    pred = ("malignant" if label else "benign") if correct else ("benign" if label else "malignant")
                    stratum = f"{'malignant' if label else 'benign'}|{'correct' if correct else 'wrong'}|" \
                              f"{'uncertain' if unc else 'confident'}"
                    pool.append({"isic_id": f"ISIC_{label}{int(correct)}{int(unc)}{j}", "label": label,
                                 "stratum": stratum, "inp": _inp(pred, unc), "correct": correct,
                                 "llm_input": {"probability_malignant": 0.81}})
    return pool


def test_full_set_is_stratified_and_codes_hide_the_stratum():
    cases = B.select_cases(_pool(), n_per_cell=5, pilot=0, seed=0)
    assert len(cases) == 40 and set(Counter(c["stratum"] for c in cases).values()) == {5}
    assert [c["case"] for c in cases] == [f"C{i:02d}" for i in range(1, 41)]
    assert len({c["stratum"] for c in cases[:8]}) > 2  # shuffled: consecutive codes are not one stratum


def test_pilot_covers_distinct_strata():
    cases = B.select_cases(_pool(), n_per_cell=5, pilot=5, seed=0)
    assert len({c["stratum"] for c in cases}) == 5


def test_counterbalancing():
    for r in range(3):
        arms = Counter(B.arm_for(i, r) for i in range(40))
        assert arms["gradcam"] == arms["text"] == 20
    assert all(len({B.arm_for(i, r) for r in range(3)}) == 2 for i in range(40))


def test_steps_lock_impression_before_model_and_explanation():
    cases = B.select_cases(_pool(), 5, 0, 0)
    steps = B.steps_for(cases, "A", 0, "en", "img")
    for code in {c["case"] for c in cases}:
        s1, s2, s3 = (next(s for s in steps if s["group"] == f"{code}s{k}") for k in (1, 2, 3))
        assert steps.index(s1) + 1 == steps.index(s2) == steps.index(s3) - 1
        assert "model predicts" not in s1["html"] and "model predicts" in s2["html"]
        assert "ISIC_" not in s1["html"] + s2["html"] + s3["html"]  # raters never see the isic_id
    fr = B.steps_for(cases, "A", 0, "fr", "img")
    assert "0,81" in next(s for s in fr if s["group"].endswith("s2"))["html"]  # French decimal comma


def test_limesurvey_tsv_blocks_mirror_each_other():
    cases = B.select_cases(_pool(), 5, 1, 0)
    steps = {lang: B.steps_for(cases, "A", 0, lang, "img") for lang in B.LANGS}
    rows = B.limesurvey_tsv(steps, "A", len(cases))
    en = [r for r in rows if r["language"] == "en"]
    fr = [r for r in rows if r["language"] == "fr"]
    assert len(en) == len(fr) and [r["class"] for r in en] == [r["class"] for r in fr]
    assert [r["id"] for r in en] == [r["id"] for r in fr]
    codes = lambda rows: [r["name"] for r in rows if r["class"] in ("Q", "A")]  # group names are translated titles
    assert codes(en) == codes(fr)
    last_en = max(i for i, r in enumerate(rows) if r["language"] == "en")
    assert last_en < min(i for i, r in enumerate(rows) if r["language"] == "fr")  # whole EN block first
    settings = {r["name"]: r["text"] for r in rows if r["class"] == "S"}
    assert settings["allowprev"] == "N" and settings["format"] == "G" and settings["additional_languages"] == "fr"
