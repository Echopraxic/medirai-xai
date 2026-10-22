"""Classifier sprint stage B: split v4, MIDAS split, diagnosis sampler, selection-score threshold."""
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import ROOT, UNC

sys.path.insert(0, str(ROOT / "splits"))
sys.path.insert(0, str(UNC))

# MIDAS-derived splits are not in git (Stanford research-use terms); rebuild them from the Redivis download
needs_midas = pytest.mark.skipif(not (ROOT / "splits/midas_v1.csv").exists() or
                                 not (ROOT / "splits/isic_v4_clinical_plus_midas.csv").exists(),
                                 reason="MIDAS splits not built (splits/make_midas_split.py, splits/make_split_v4.py)")


@needs_midas
def test_midas_split_keeps_patients_together_and_labels_isic_consistent():
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    use = m[m.split != "excluded"]
    assert (use.groupby("patient_id").split.nunique() == 1).all()
    assert set(use.split) == {"train", "dev_val", "dev_test"}
    assert not use.midas_path.str.contains("ak|re-excision", regex=True).any()   # ISIC "Indeterminate"


@needs_midas
def test_split_v4_adds_only_midas_phone_photos_without_leaks():
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    v4 = pd.read_csv(ROOT / "splits/isic_v4_clinical_plus_midas.csv")
    assert v4[v4.source != "MIDAS"].drop(columns="modality").reset_index(drop=True).equals(v2)  # v2 rows unchanged
    mid = v4[v4.source == "MIDAS"]
    assert mid.modality.str.startswith("phone").all() and mid.diagnosis_3.notna().all()
    assert set(mid.split) == {"train", "val", "dev_test"}
    for a, b in (("train", "val"), ("train", "dev_test"), ("val", "dev_test")):
        assert not set(mid[mid.split == a].group_id) & set(mid[mid.split == b].group_id)


def test_diagnosis_sampler_favours_rare_diagnoses_reproducibly():
    import dataset
    df = pd.DataFrame({"diagnosis_3": ["Basal cell carcinoma"] * 90 + ["Seborrheic keratosis"] * 10})
    draw = lambda: np.fromiter(iter(dataset.diagnosis_sampler(df, 0.5, torch.Generator().manual_seed(0))), int)  # noqa: E731
    idx = draw()
    assert len(idx) == len(df)                                  # epoch length unchanged
    share_sk = (idx >= 90).mean()
    assert abs(share_sk - 0.25) < 0.08                          # sqrt(90) / (sqrt(90) + sqrt(10)) = 0.25 vs 0.10 plain
    assert (idx == draw()).all()


def test_threshold_for_sensitivity_hits_target():
    sys.path.insert(0, str(ROOT / "eval"))
    from selection_score import threshold_for_sensitivity
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(100)].astype(int)
    p = np.r_[rng.uniform(0.3, 1, 200), rng.uniform(0, 0.7, 100)]
    t = threshold_for_sensitivity(y, p)
    assert (p[y == 1] >= t).mean() >= 0.90 and (p[y == 1] >= t + 1e-9).mean() < 0.90 + 0.01


def test_derm7pt_labels_follow_isic_conventions():
    from make_split_v5 import d7_label
    assert d7_label("melanoma (in situ)") == ("Melanoma in situ", 1)
    assert d7_label("melanoma (less than 0.76 mm)") == ("Melanoma Invasive", 1)
    assert d7_label("melanoma") == ("Melanoma, NOS", 1)
    assert d7_label("basal cell carcinoma") == ("Basal cell carcinoma", 1)
    assert d7_label("reed or spitz nevus") == ("Nevus", 0) and d7_label("clark nevus") == ("Nevus", 0)
    assert d7_label("seborrheic keratosis") == ("Seborrheic keratosis", 0)
    assert d7_label("miscellaneous") == (None, None)


@pytest.mark.skipif(not (ROOT / "splits/isic_v6_v4_plus_melanoma.csv").exists(), reason="split v6 not built")
def test_split_v6_adds_melanoma_data_to_train_only():
    v4 = pd.read_csv(ROOT / "splits/isic_v4_clinical_plus_midas.csv")
    v6 = pd.read_csv(ROOT / "splits/isic_v6_v4_plus_melanoma.csv")
    new = v6[~v6.isic_id.isin(v4.isic_id)]
    assert set(new.split) == {"train"} and set(new.source) == {"Derm7pt", "Asan", "SNU"}
    for s in ("val", "test", "dev_test"):     # selection and test rows unchanged
        assert set(v6.loc[v6.split == s, "isic_id"]) == set(v4.loc[v4.split == s, "isic_id"])
