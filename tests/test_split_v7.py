"""Split v7 / darker-skin dev set builder (splits/make_split_v7.py, prereg 2026-10-23): task mapping and dedupe."""
import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "splits"))


def test_f17k_task_mapping_follows_the_prereg():
    import make_split_v7 as M
    df = pd.DataFrame({"md5hash": list("abcdefgh"), "fitzpatrick_scale": [5, 6, 5, 4, 3, 6, 5, 5],
                       "label": ["melanoma", "seborrheic keratosis", "kaposi sarcoma", "lentigo maligna", "melanoma",
                                 "prurigo nodularis", "psoriasis", "squamous cell carcinoma"],
                       "three_partition_label": ["malignant", "benign", "malignant", "malignant", "malignant", "benign",
                                                 "non-neoplastic", "malignant"]})
    t = M.f17k_task(df).set_index("md5hash")
    assert list(t.index) == ["a", "b", "d", "h"]        # kaposi, FST 3, prurigo and non-neoplastic dropped
    assert t.y.to_dict() == {"a": 1, "b": 0, "d": 1, "h": 1}
    assert "Melanoma" in t.loc["d", "diagnosis_3"]       # lentigo maligna counts as melanoma for the melanoma head


def test_min_distance_and_groups_within():
    import make_split_v7 as M
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, (1, 256)).astype(np.float32)
    near = a.copy()
    near[0, :5] = 1 - near[0, :5]                         # 5 bits away
    far = 1 - a                                            # 256 bits away
    assert list(M.min_distance(np.vstack([near, far]), a)) == [5, 256]
    assert M.groups_within(np.vstack([a, near, far]), ["x", "y", "z"]) == ["x", "x", "z"]
