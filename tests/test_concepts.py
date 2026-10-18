import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "explain"))
sys.path.insert(0, str(ROOT / "concepts"))


def test_derm7pt_image_lookup_ignores_case(tmp_path):
    from vlm_concepts import find_image
    (tmp_path / "FCL").mkdir()
    (tmp_path / "FCL" / "Fcl068.jpg").write_bytes(b"x")
    # meta.csv says "FCl/..."; the folder on disk is "FCL" (fails on Linux without the case-insensitive match)
    assert find_image(tmp_path, "FCl/Fcl068.jpg").name.lower() == "fcl068.jpg"
    with pytest.raises(FileNotFoundError):
        find_image(tmp_path, "FCl/missing.jpg")


def test_phrase_bank_status_rules():
    sys.path.insert(0, str(ROOT / "concepts"))
    from phrase_bank import judge
    enh = {"concepts": {"asymmetry": {"single_feature_rho": {"A_x": 0.40, "A_y": 0.10}, "crowd_rater_vs_others_rho": 0.38,
                                      "labels_usable": True},
                        "border": {"single_feature_rho": {"B_x": 0.9}, "crowd_rater_vs_others_rho": 0.01,
                                   "labels_usable": False}}}
    assert judge(("enhance", "asymmetry"), "A_x", enh, None, None)[0] == "validated"   # at least rater level
    assert judge(("enhance", "asymmetry"), "A_y", enh, None, None)[0] == "failed"
    assert judge(("enhance", "border"), "B_x", enh, None, None)[0] == "untestable"     # unreliable labels win
    monet = {"pairs": {"MONET_erythema~C_r": 0.07}}
    assert judge(("monet", "MONET_erythema~C_r"), "C_r", None, monet, None)[0] == "failed"
    assert judge(("derm7pt", "elevation"), "T_r", None, None, {"elevation": {"T_r": 0.64}})[0] == "partial"
