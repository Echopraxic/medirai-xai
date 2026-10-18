import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))


def test_results_page_builds_from_whatever_outputs_exist(tmp_path):
    import make_results
    out = tmp_path / "RESULTS.md"
    make_results.main(type("Opts", (), {"out": str(out)})())
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Results (generated, do not edit)")
    assert "## Success criteria" in text and "C7 clinician review" in text
    assert "## D5 automated validation" in text


def test_c1_met_does_not_claim_overall_pass_below_target():
    from make_results import c1_met
    assert c1_met({"overall": 0.843, "by_true_label": {"0": 0.72, "1": 0.89}}) == "no"
    assert c1_met({"overall": 0.858, "by_true_label": {"0": 0.73, "1": 0.91}}) == "overall yes, per class no"
    assert c1_met({"overall": 0.90, "by_true_label": {"0": 0.86, "1": 0.92}}) == "yes"
