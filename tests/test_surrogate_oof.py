import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "splits"))


def test_oof_folds_keep_groups_whole_and_cover_train_only():
    from make_oof_v8f import K, folds
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"isic_id": [f"I{i}" for i in range(n)], "group_id": [f"G{i // 2}" for i in range(n)],
                       "label": rng.integers(0, 2, n), "split": ["train"] * 150 + ["val"] * 30 + ["test"] * 20})
    f = folds(df)
    assert set(f) == set(df[df.split == "train"].isic_id)                    # val/test never get a fold
    assert set(f.values()) == set(range(K))
    g = df[df.split == "train"].assign(fold=lambda d: d.isic_id.map(f))
    assert (g.groupby("group_id").fold.nunique() == 1).all()                # a lesion never straddles folds


def test_extra_rows_fail_flag_without_sam():
    from eval.surrogate_c1b import fail_without_sam
    area = pd.Series([0.001, 0.1, 0.9, 0.1])
    prob = pd.Series([0.95, 0.95, 0.95, 0.7])
    assert fail_without_sam(area, prob).tolist() == [True, False, True, True]

