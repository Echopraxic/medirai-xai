"""Sprint part 3 analysis (eval/sprint3_analysis.py, prereg 2026-10-30): run discovery, DSD v2 mask, learning-curve slope."""
import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "eval"))


def test_find_runs_matches_one_arm_only(tmp_path, monkeypatch):
    import sprint3_analysis as S
    for name in ("narval_stageb_kdv6_mel_s0_20261022", "narval_stageb_kdv6_mel_cons_s0_20261023",
                 "narval_stageb_kdv8f_mel_s3_20261031", "narval_stageb_kdv8f_lc_k1_n2_s0_20261031"):
        (tmp_path / "output" / name / "run").mkdir(parents=True)
        (tmp_path / "output" / name / "run" / "tta_scores.npz").write_bytes(b"")
    monkeypatch.setattr(S, "UNC", tmp_path)
    assert S.find_runs("kdv6_mel") == {0: "narval_stageb_kdv6_mel_s0_20261022/run"}
    assert S.find_runs("kdv8f_mel") == {3: "narval_stageb_kdv8f_mel_s3_20261031/run"}
    assert list(S.find_runs("kdv8f_lc_k1_n2")) == [0]


def test_dsd_v2_mask_and_slope():
    import sprint3_analysis as S
    M = {"dsd": pd.DataFrame({"label": [0, 1, 1], "dsd": [True, True, False]}, index=["a", "b", "c"])}
    out = S.dsd_v2_meta(M, pd.DataFrame({"isic_id": ["b", "c"]}))
    assert list(out["dsd"].dsd) == [False, True, True] and list(M["dsd"].dsd) == [True, True, False]
    assert np.isclose(S.slope([0, 3, 7], [0.60, 0.64, 0.66]), 0.02)        # log2(1 + added) = 0, 2, 3
