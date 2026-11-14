import numpy as np
import pandas as pd

from interpret.passion_labels import build, excluded


def test_excluded_body_sites():
    assert excluded("torso; pelvic; arm")
    assert excluded("pelvic")
    assert excluded(np.nan) and excluded("") and excluded("  ")
    assert not excluded("hair; face; arm")


def _labels(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"subject_id": [f"S{i:03d}" for i in range(n)],
                         "fitzpatrick": rng.choice([3, 4, 5, 6], n),
                         "body_loc": ["arm"] * (n - 2) + ["leg; pelvic", np.nan]})


def test_build_drops_excluded_subjects_and_keeps_subjects_whole():
    lab = _labels()
    ids = [f"{s}_{k}" for s in lab.subject_id for k in (1, 2)]
    out, dups = build(lab, ids)
    assert not out.subject_id.isin({"S038", "S039"}).any()
    assert len(out) == 2 * 38 and dups.empty
    assert (out.groupby("subject_id").split.nunique() == 1).all()
    assert set(out.split) == {"train", "val", "test"}
    assert (out.dark == (out.fitzpatrick >= 5)).all()
    assert not {"age", "sex", "country", "body_loc"} & set(out.columns)


def test_near_duplicates_join_subjects_into_one_split():
    lab = _labels()
    ids = [f"{s}_1" for s in lab.subject_id[:38]]
    emb = np.random.default_rng(1).normal(size=(len(ids), 8)).astype(np.float32)
    emb[5] = emb[20] + 1e-4  # S005 and S020: same photo
    out, dups = build(lab, ids, ids, emb)
    assert len(dups) == 1
    r = out.set_index("subject_id")
    assert r.loc["S005", "subject_group"] == r.loc["S020", "subject_group"]
    assert r.loc["S005", "split"] == r.loc["S020", "split"]
