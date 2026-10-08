"""Guards for the persisted ISIC clinical split (CODEBASE_TODO P0-1/P0-2/P0-3)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splits"))
from make_isic_clinical_split import LABEL_MAP, connected_groups, leakage, make_split  # noqa: E402

SPLIT = ROOT / "splits" / "isic_clinical_v1.csv"
META = ROOT / "splits" / "isic_clinical_closeup_metadata.csv"


@pytest.fixture(scope="module")
def split():
    return pd.read_csv(SPLIT, dtype={"label": "Int64"})


def test_ids_unique_and_complete(split):
    meta = pd.read_csv(META, low_memory=False)
    assert split["isic_id"].is_unique
    assert set(split["isic_id"]) == set(meta["isic_id"])


def test_no_lesion_or_patient_crosses_splits(split):
    used = split[split["split"] != "excluded"]
    for key in ("group_id", "lesion_id", "patient_id"):
        per_key = used.dropna(subset=[key]).groupby(key)["split"].nunique()
        assert (per_key == 1).all(), f"{key} spans multiple splits: {per_key[per_key > 1].index[:5].tolist()}"
    assert sum(n for n, _ in leakage(split).values()) == 0


def test_excluded_is_exactly_non_binary_labels(split):
    binary = split["diagnosis_1"].isin(LABEL_MAP)
    assert (split.loc[~binary, "split"] == "excluded").all()
    assert (split.loc[binary, "split"] != "excluded").all()
    assert split.loc[binary, "label"].notna().all()


def test_proportions_and_balance(split):
    used = split[split["split"] != "excluded"]
    share = used["split"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.02)
    assert share["val"] == pytest.approx(0.10, abs=0.02)
    assert share["test"] == pytest.approx(0.20, abs=0.02)
    malignant = used.groupby("split")["label"].mean()
    assert malignant.max() - malignant.min() < 0.05


def test_split_is_reproducible(split):
    rebuilt = make_split(pd.read_csv(META, low_memory=False))
    pd.testing.assert_series_equal(
        rebuilt.set_index("isic_id")["split"].sort_index(),
        split.set_index("isic_id")["split"].sort_index())


def test_connected_groups_merges_through_shared_patient():
    df = pd.DataFrame({"lesion_id": ["L1", "L1", "L2", None, None],
                       "patient_id": [None, "P1", "P1", None, "P9"]})
    g = connected_groups(df)
    assert g[0] == g[1] == g[2]          # L1 ~ P1 ~ L2
    assert len({g[0], g[3], g[4]}) == 3  # unlinked images stay separate


def test_v2_only_excludes_mskcc_and_never_moves_images(split):
    """isic_clinical_v2 (2026-10-07) = v1 with MSKCC excluded; no image may change train/val/test."""
    from make_split_v2 import build_v2
    meta = pd.read_csv(META, low_memory=False)
    v2 = build_v2(split, meta)
    on_disk = pd.read_csv(ROOT / "splits" / "isic_clinical_v2.csv")
    assert v2[["isic_id", "split"]].equals(on_disk[["isic_id", "split"]])
    both = split.merge(v2, on="isic_id", suffixes=("_v1", "_v2"))
    moved = both[(both.split_v1 != both.split_v2)]
    assert (moved.split_v2 == "excluded").all()
    assert (moved.attribution_v1 == "Memorial Sloan Kettering Cancer Center").all()
    assert not ((v2.source == "MSKCC") & (v2.split != "excluded")).any()
    assert (v2.loc[v2.split == "excluded", "exclusion_reason"] != "").all()
