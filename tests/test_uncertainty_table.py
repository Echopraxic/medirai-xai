"""eval/uncertainty_table.py (S1-4): per-image uncertainty, agreement and deferral across models."""
import sys

import pandas as pd
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "eval"))
from uncertainty_table import build  # noqa: E402

IDS = [f"ISIC_{i:07d}" for i in range(8)]
SPLIT = ["val"] * 4 + ["test"] * 4
LABEL = [0, 1, 0, 1, 0, 1, 0, 1]


def _csv(tmp_path, name, pred, entropy=None, mc_std=None):
    d = pd.DataFrame({"isic_id": IDS, "split": SPLIT, "label": LABEL, "pred": pred,
                      "p_malignant": [0.9 if p else 0.1 for p in pred]})
    if entropy is not None:
        d["entropy_of_expected"] = entropy
        d["expected_entropy"] = [e / 2 for e in entropy]
        d["mutual_information"] = [e / 2 for e in entropy]
    if mc_std is not None:
        d["mc_std_p_malignant"] = mc_std
    path = tmp_path / f"{name}.csv"
    d.to_csv(path, index=False)
    return path


@pytest.fixture
def runs(tmp_path):
    ent = [0.1, 0.2, 0.3, 0.9, 0.1, 0.95, 0.2, 0.1]   # val 75th pct = 0.45 -> test image 5 is high entropy
    return {"det": _csv(tmp_path, "det", LABEL, ent, [0.0] * 8),
            "vll": _csv(tmp_path, "vll", [0, 1, 0, 1, 1, 1, 0, 1], ent, [0.1] * 8),
            "tree": _csv(tmp_path, "tree", [0, 1, 0, 1, 1, 1, 1, 1])}


def test_disagreement_and_entropy_drive_deferral(runs):
    df, summary = build(runs, "det")
    t = df.set_index("isic_id")
    assert t.loc["ISIC_0000004", "disagreeing_models"] == "vll,tree"
    assert t.loc["ISIC_0000006", "disagreeing_models"] == "tree"
    assert t.loc["ISIC_0000005", "defer_reason"] == "high entropy"
    assert t.loc["ISIC_0000004", "defer_reason"] == "models disagreed (vll,tree)"
    assert not t.loc["ISIC_0000007", "deferred"] and t.loc["ISIC_0000007", "defer_reason"] == ""
    assert summary["test_deferred_fraction"] == 0.75
    assert summary["test_primary_accuracy"]["deferred"] == 1.0  # det is right everywhere in this toy set


def test_variational_variance_only_for_mc_runs(runs):
    df, _ = build(runs, "det")
    assert "vll_variational_variance" in df and df.vll_variational_variance.to_numpy() == pytest.approx(0.01)
    assert "det_variational_variance" not in df and "tree_entropy_of_expected" not in df


def test_primary_needs_entropy(runs):
    with pytest.raises(ValueError, match="no entropy_of_expected"):
        build(runs, "tree")
