"""Split v8 / DSD v2 builder (splits/make_split_v8.py, prereg part 3 2026-10-30): cleaning, mapping, learning curve."""
import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "splits"))


def _dsd_v1(n=12):
    return pd.DataFrame({"isic_id": [f"F17K_{i:02d}" for i in range(n)], "label": [i % 2 for i in range(n)],
                         "group_id": [f"F17K_{i // 2 * 2:02d}" for i in range(n)], "diagnosis_3": "x",
                         "dsd": [i != 0 for i in range(n)]})


def test_dsd_v2_keeps_fst56_rows_cleaned_by_f17k_c_with_group_folds():
    import make_split_v8 as M
    c = pd.DataFrame({"md5hash": [f"{i:02d}" for i in range(12) if i != 3],
                      "qc": [("3 Wrongly labelled" if i == 5 else np.nan) for i in range(12) if i != 3]})
    d = M.dsd_v2(_dsd_v1(), c)
    assert set(d.isic_id) == {f"F17K_{i:02d}" for i in (1, 2, 4, 6, 7, 8, 9, 10, 11)}   # FST 4 (0), not in C (3), wrong (5)
    assert d.groupby("group_id").fold.nunique().max() == 1 and set(d.fold) == {0, 1, 2}


def test_f17k_c_task_uses_corrected_labels_and_fst_range():
    import make_split_v8 as M
    c = pd.DataFrame({"md5hash": list("abcde"), "fst": [1, 4, 5, 2, 0],
                      "label": ["melanoma", "seborrheic keratosis", "melanoma", "mycosis fungoides", "melanoma"],
                      "three_partition_label": ["malignant", "benign", "malignant", "malignant", "malignant"]})
    t = M.f17k_c_task(c, range(1, 5)).set_index("md5hash")
    assert list(t.index) == ["a", "b"] and t.loc["a", "y"] == 1 and t.loc["b", "y"] == 0   # FST 5, MF, unknown FST out


def test_learning_curve_adds_other_folds_and_holds_out_fold_k():
    import make_split_v8 as M
    dsd = _dsd_v1().assign(fold=[i % 3 for i in range(12)])
    base = pd.DataFrame({"isic_id": ["ISIC_1"], "split": ["val"]})
    for k in range(3):
        for n in (1, 2):
            s = M.lc_split(base, dsd, k, n)
            added = set(s.isic_id) - {"ISIC_1"}
            assert added == set(dsd[dsd.fold.isin([(k + i) % 3 for i in range(1, n + 1)])].isic_id)
            assert not added & set(dsd[dsd.fold == k].isic_id) and (s.split[s.isic_id.isin(added)] == "train").all()


def test_sd198_mapping_follows_split_v2_malignancy():
    import make_split_v8 as M
    assert M.SD198_MALIGNANT["Bowens_Disease"] == "Squamous cell carcinoma in situ"
    assert "Keratoacanthoma" in M.SD198_MALIGNANT and not set(M.SD198_MALIGNANT) & set(M.SD198_BENIGN)
    assert not any("Actinic" in k or "Lymphoma" in k for k in {**M.SD198_MALIGNANT, **M.SD198_BENIGN})
