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
