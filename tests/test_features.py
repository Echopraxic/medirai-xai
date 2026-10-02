"""
Feature extraction v0 on synthetic lesions with known properties (plan T1.4): each feature must move in the
direction its name and dictionary entry claim.
"""
import sys

import numpy as np
import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "features"))
import extract as fx  # noqa: E402

SKIN = (215, 170, 150)
BROWN = (180, 125, 85)
BLACK = (30, 30, 30)


def canvas(h=400, w=500):
    yy, xx = np.mgrid[:h, :w]
    return yy, xx


def render(mask, fill=BROWN, skin=SKIN, noise=0.0, seed=0):
    img = np.empty(mask.shape + (3,), dtype=float)
    img[:] = skin
    img[mask] = fill
    if noise:
        img[mask] += np.random.default_rng(seed).normal(0, noise, (mask.sum(), 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def ellipse(a, b, angle=0.0, h=400, w=500, cy=200, cx=250):
    yy, xx = canvas(h, w)
    t = np.radians(angle)
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def star(r=110, amp=0.25, k=9, h=400, w=500):
    yy, xx = canvas(h, w)
    theta = np.arctan2(yy - 200, xx - 250)
    return np.hypot(yy - 200, xx - 250) <= r * (1 + amp * np.sin(k * theta))


def lopsided():
    m = ellipse(110, 110)
    yy, xx = canvas()
    m |= (np.hypot(yy - 150, xx - 360) <= 60)   # a lobe on one side only
    return m


@pytest.fixture(scope="module")
def disc():
    m = ellipse(110, 110)
    return fx.extract(render(m), m)


def test_disc_is_symmetric_round_and_regular(disc):
    assert disc["A_shape_asym_max"] < 0.05
    assert disc["B_circularity"] > 0.85
    assert disc["B_convexity"] > 0.95
    assert disc["B_radial_cv"] < 0.03
    assert disc["S_eccentricity"] < 0.2 and disc["S_axis_ratio"] > 0.95


def test_rotated_ellipse_is_symmetric_but_elongated():
    m = ellipse(150, 70, angle=33)
    f = fx.extract(render(m), m)
    assert f["A_shape_asym_max"] < 0.06          # alignment to the principal axes works for any rotation
    assert f["S_eccentricity"] > 0.85 and f["S_axis_ratio"] < 0.55


def test_lopsided_lesion_is_asymmetric(disc):
    m = lopsided()
    f = fx.extract(render(m), m)
    assert f["A_shape_asym_max"] > disc["A_shape_asym_max"] + 0.15


def test_star_border_is_irregular(disc):
    m = star()
    f = fx.extract(render(m), m)
    assert f["B_circularity"] < disc["B_circularity"] - 0.2
    assert f["B_convexity"] < 0.95
    assert f["B_radial_cv"] > disc["B_radial_cv"] + 0.1
    assert f["B_fractal_dim"] > disc["B_fractal_dim"]


def test_two_color_lesion_counts_two_colors_and_is_color_asymmetric(disc):
    m = ellipse(110, 110)
    img = render(m)
    yy, xx = canvas()
    img[m & (xx < 250)] = BLACK
    f = fx.extract(img, m)
    assert disc["C_n_colors"] == 1 and disc["C_frac_light_brown"] > 0.9
    assert f["C_n_colors"] == 2 and f["C_frac_black"] == pytest.approx(0.5, abs=0.05)
    assert f["A_color_asym_deltaE"] > disc["A_color_asym_deltaE"] + 20
    assert f["C_color_entropy"] > disc["C_color_entropy"]


def test_dark_and_red_relative_to_skin():
    m = ellipse(110, 110)
    dark = fx.extract(render(m, fill=BLACK), m)
    red = fx.extract(render(m, fill=(200, 90, 90)), m)
    assert dark["C_darkness_vs_skin"] > 40
    assert red["C_redness_vs_skin"] > dark["C_redness_vs_skin"]


def test_diameter_is_relative_to_the_frame():
    m = ellipse(100, 100)
    f = fx.extract(render(m), m)
    assert f["D_feret_rel"] == pytest.approx(200 / np.hypot(400, 500), rel=0.05)


def test_rough_surface_has_more_texture(disc):
    m = ellipse(110, 110)
    f = fx.extract(render(m, noise=25), m)
    assert f["T_roughness"] > disc["T_roughness"] + 2
    assert f["T_glcm_contrast"] > disc["T_glcm_contrast"]
    assert f["T_glcm_homogeneity"] < disc["T_glcm_homogeneity"]


def scaled_lesion(k):
    """Elongated, lopsided, notched lesion drawn at k x the base resolution (500 x 400)."""
    yy, xx = np.mgrid[:400 * k, :500 * k] / k
    theta = np.arctan2(yy - 200, xx - 250)
    body = ((xx - 250) / 150) ** 2 + ((yy - 200) / 90) ** 2 <= (1 + 0.12 * np.sin(7 * theta)) ** 2
    lobe = np.hypot(yy - 140, xx - 330) <= 45
    return body | lobe


def test_features_do_not_depend_on_source_resolution():
    """Same lesion at 500 px and 2000 px wide (ISIC sources differ by resolution) -> near-identical features."""
    small, large = scaled_lesion(1), scaled_lesion(4)
    fs, fl = fx.extract(render(small), small), fx.extract(render(large), large)
    for k in ("A_shape_asym_major", "A_shape_asym_minor", "B_circularity", "B_convexity", "B_radial_cv",
              "B_fractal_dim", "C_n_colors", "D_feret_rel", "S_eccentricity", "S_solidity", "S_axis_ratio"):
        assert fl[k] == pytest.approx(fs[k], rel=0.1, abs=0.02), k


def test_every_feature_is_in_the_dictionary(disc):
    assert set(disc) == set(fx.FEATURE_DICTIONARY)
    assert {v[0] for v in fx.FEATURE_DICTIONARY.values()} <= set(fx.CONCEPT_GROUPS)
    assert all(k.split("_")[0] == v[0] for k, v in fx.FEATURE_DICTIONARY.items())
