import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "eval"))


def test_milk_scale_is_dark_at_zero_and_takes_precedence_over_fitzpatrick():
    from skin_tone_report import coarse_group
    assert coarse_group(0, None) == "darker" and coarse_group(2, None) == "darker"
    assert coarse_group(3, None) == "medium" and coarse_group(5, None) == "lighter"
    assert coarse_group(np.nan, "VI") == "darker" and coarse_group(np.nan, "III") == "medium"
    assert coarse_group(np.nan, "I") == "lighter" and coarse_group(np.nan, np.nan) == "unknown"
    assert coarse_group(5, "VI") == "lighter"   # one scale per image: MILK first


def test_tone_table_and_metrics():
    from skin_tone_report import tone_table, metrics
    split = pd.DataFrame({"isic_id": ["a", "b", "c"], "split": "test", "label": [1, 0, 1], "group_id": [1, 2, 3],
                          "source": ["MILK", "UFES", "UFES"]})
    milk = pd.DataFrame({"isic_id": ["a"], "skin_tone_class": [1]})
    fitz = pd.DataFrame({"isic_id": ["b", "c"], "fitzpatrick_skin_type": ["II", None]})
    t = tone_table(split, milk, fitz)
    assert t.set_index("isic_id").tone_group.to_dict() == {"a": "darker", "b": "lighter", "c": "unknown"}
    m = metrics(np.array([1, 0, 1, 0]), np.array([.9, .2, .4, .6]), np.array([1, 0, 0, 1]))
    assert m["accuracy"] == 0.5 and m["malignant_recall"] == 0.5 and m["auroc"] == 0.75
