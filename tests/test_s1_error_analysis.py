"""S1-9 analysis scripts: error strata, lesion-level bootstrap, feature-validity wording, TreeSHAP contrasts."""
import sys

import numpy as np
import pandas as pd

from conftest import ROOT

sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "trees"))
import s1_error_lists as el  # noqa: E402
import s1_intervals_figures as fig  # noqa: E402
from feature_validity import best_status, wording  # noqa: E402
from s1_error_drivers import contrast  # noqa: E402


def test_entropy_matches_binary_nats():
    assert np.isclose(el.entropy([0.5])[0], np.log(2))
    assert el.entropy([0.0, 1.0]).max() < 1e-9


def test_stratum_assignment():
    rows = pd.DataFrame({"pred": [1, 0, 1, 0], "label": [1, 1, 1, 1], "deferred": [False, False, True, True]})
    assert [el.stratum(r) for r in rows.itertuples()] == ["confident_right", "confident_wrong",
                                                          "deferred_right", "deferred_wrong"]


def _strata(n_lesions=40, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_lesions), 2)
    label = np.repeat(rng.integers(0, 2, n_lesions), 2)
    pred = np.where(rng.random(len(g)) < 0.8, label, 1 - label)
    return pd.DataFrame({"group": g, "label": label, "pred": pred, "melanoma": False,
                         "deferred": rng.random(len(g)) < 0.3, "entropy_of_expected": rng.random(len(g))})


def test_rates_point_values():
    d = pd.DataFrame({"group": [0, 1, 2, 3], "label": [1, 1, 0, 0], "pred": [1, 0, 0, 1], "melanoma": False,
                      "deferred": [True, False, False, True], "entropy_of_expected": [0.6, 0.1, 0.2, 0.5]})
    r = fig.rates(d)
    assert r["sensitivity"] == 0.5 and r["specificity"] == 0.5 and r["share deferred"] == 0.5
    assert r["error rate, deferred"] == 0.5 and np.isnan(r["melanoma recall"])


def test_cluster_bootstrap_interval_contains_point_and_is_reproducible():
    d = _strata()
    a, b = fig.cluster_bootstrap(d, n_boot=200), fig.cluster_bootstrap(d, n_boot=200)
    assert a == b
    v, lo, hi = a["sensitivity"]
    assert lo <= v <= hi and hi - lo > 0


def test_best_status_prefers_validated_then_auroc():
    val = {"confirmatory": [
        {"candidate": "C_x", "kind": "handcrafted", "status": "partial", "concept": "a", "dataset": "d", "auroc": 0.7,
         "diagnosis_confounded": False},
        {"candidate": "C_x", "kind": "handcrafted", "status": "validated", "concept": "b", "dataset": "d",
         "auroc": 0.8, "diagnosis_confounded": False},
        {"candidate": "M_y", "kind": "MONET", "status": "validated", "concept": "c", "dataset": "d", "auroc": 0.9,
         "diagnosis_confounded": False}]}
    s = best_status(val)
    assert list(s.index) == ["C_x"] and s.loc["C_x", "concept"] == "b"
    assert wording(s.loc["C_x"]).startswith("b (validated")


def test_contrast_ranks_the_feature_that_separates_groups():
    d = pd.DataFrame({"isic_id": [f"i{k}" for k in range(12)], "outcome": ["FN"] * 6 + ["TP"] * 6,
                      "C_a": [-1.0] * 6 + [1.0] * 6, "C_b": [0.0] * 12})
    values = d.set_index("isic_id")[["C_a", "C_b"]] * 10
    t, na, nb = contrast(d, ["C_a", "C_b"], values, {"C_a": "partial", "C_b": "failed"},
                         pd.Series(True, index=d.index), "FN", "TP")
    assert (na, nb) == (6, 6) and t.iloc[0].feature == "C_a" and t.iloc[0].shap_difference == -2.0


# --- interpret/scin_labels.py ---------------------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "interpret"))
import scin_labels as scin  # noqa: E402


def test_efst_median_tie_and_agreement():
    lab = pd.DataFrame({"case_id": [1, 2, 3, 4],
                        **{f"dermatologist_fitzpatrick_skin_type_label_{i}": v for i, v in
                           zip((1, 2, 3), (["FST5", "FST4", "FST6", np.nan], ["FST5", "FST5", np.nan, np.nan],
                                           ["FST6", np.nan, np.nan, np.nan]))},
                        **{f"dermatologist_gradable_for_fitzpatrick_skin_type_{i}": v for i, v in
                           zip((1, 2, 3), (["YES", "YES", "YES", "NO"], ["YES", "YES", None, None],
                                           ["YES", None, None, None]))}})
    e = scin.efst_labels(lab).set_index("case_id")
    assert e.loc[1, "efst"] == 5 and e.loc[1, "dark_agreed"] and e.loc[1, "n_efst_raters"] == 3
    assert e.loc[2, "efst_tie"] and e.loc[2, "dark"] == 0          # 4 and 5 -> 4.5 counts as not dark
    assert e.loc[3, "dark"] == 1 and not e.loc[3, "dark_agreed"]    # one rater only
    assert np.isnan(e.loc[4, "dark"])                               # ungradable


def test_case_groups_join_cases_that_share_an_image():
    cases = pd.DataFrame({"case_id": [1, 2, 3], "image_1_path": ["a.png", "b.png", "c.png"],
                          "image_2_path": [np.nan, "a.png", np.nan], "image_3_path": [np.nan] * 3})
    g = scin.case_groups(cases)
    assert g[0] == g[1] and g[2] != g[0]
