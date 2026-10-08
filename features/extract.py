"""
Feature extraction: concept-aligned handcrafted lesion features (A/B/C, texture, shape).

v0.2 (2026-10-07) normalizes away the camera before measuring the lesion. On v0.1 the features identified the
image source within a single diagnosis (BCC, nevus, SK: MILK vs UFES AUROC 0.95-0.97, mostly texture and
color), because sources differ in resolution (MILK is fixed at 600 px; UFES 147-3,476 px), framing (UFES
lesions fill ~2x more of the frame) and white balance. v0.2 therefore:
  1. applies Shades-of-Gray color constancy (Finlayson & Trezzi 2004, p=6; standard for skin images) to the
     whole photo;
  2. crops around the lesion and resamples it so its max Feret diameter is LESION_SIDE px, so texture and
     the skin ring are measured at the same scale per lesion, whatever the camera or framing;
  3. moves the frame-relative size measures out of the concept groups (`frame_feret_rel`, `frame_area_rel`):
     they describe how the photo was framed, not the lesion, so they are kept for QA but never modelled
     or explained. There is no D (mm) feature: no calibration exists outside UFES.

Feature names are <GROUP>_<name>; GROUP is the concept group used for SHAP grouping:
  A asymmetry · B border · C color · T texture/surface · S shape
See FEATURE_DICTIONARY for units and direction. Thresholds that turn these into clinical words are NOT
here (concepts/, WS3).
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color, feature, measure

FEATURE_VERSION = "v0.2"
LESION_SIDE = 192   # lesion max Feret diameter after resampling (px); MILK lesions are ~175 px natively
CROP_MARGIN = 0.35  # context kept around the lesion, as a fraction of its Feret diameter (skin ring needs ~0.2)
SOG_P = 6           # Shades-of-Gray Minkowski norm

# Reference colors for the dermoscopic ABCD color criterion (sRGB). Literature prototypes, not calibrated
# for clinical photographs; C_n_colors and C_frac_* are therefore exploratory until WS3 validation.
COLOR_PROTOTYPES_RGB = {
    "white": (235, 235, 235),
    "red": (190, 60, 60),
    "light_brown": (180, 125, 85),
    "dark_brown": (95, 60, 40),
    "blue_gray": (100, 120, 150),
    "black": (30, 30, 30),
}
COLOR_MIN_FRACTION = 0.05  # a named color "is present" if it covers >= 5% of the lesion


def shades_of_gray(img: np.ndarray, p: int = SOG_P) -> np.ndarray:
    """Color constancy: divide out the illuminant estimated as the per-channel Minkowski p-norm mean."""
    x = img.astype(float) / 255.0
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    gain = illum.mean() / np.maximum(illum, 1e-6)
    return (np.clip(x * gain, 0, 1) * 255).round().astype(np.uint8)


def frame_features(mask: np.ndarray) -> dict:
    """How the photo is framed (QA only; not a lesion property, never used by the models)."""
    props = measure.regionprops(mask.astype(np.uint8))[0]
    return {"frame_feret_rel": float(props.feret_diameter_max / np.hypot(*mask.shape)),
            "frame_area_rel": float(mask.mean())}


def to_lesion_scale(img: np.ndarray, mask: np.ndarray):
    """Crop around the lesion (with skin context) and resample so its max Feret diameter is LESION_SIDE px."""
    props = measure.regionprops(mask.astype(np.uint8))[0]
    feret = max(props.feret_diameter_max, 1.0)
    pad = int(np.ceil(CROP_MARGIN * feret))
    y0, x0, y1, x1 = props.bbox
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(mask.shape[0], y1 + pad), min(mask.shape[1], x1 + pad)
    img, mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    scale = LESION_SIDE / feret
    size = (max(1, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale)))
    img = np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS if scale < 1 else Image.BICUBIC))
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return img, mask


def _principal_frame(mask: np.ndarray):
    """Centroid and unit vectors of the major/minor axes (eigenvectors of the pixel covariance)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    vals, vecs = np.linalg.eigh(cov)
    major, minor = vecs[:, 1], vecs[:, 0]          # (dx, dy)
    return cy, cx, major, minor, ys, xs


def _mask_in_principal_frame(mask: np.ndarray) -> np.ndarray:
    """Resample the mask on a (v, u) grid where u runs along the major axis and v along the minor axis."""
    cy, cx, major, minor, ys, xs = _principal_frame(mask)
    r = int(np.ceil(np.hypot(ys - cy, xs - cx).max())) + 2
    v, u = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    x = cx + u * major[0] + v * minor[0]
    y = cy + u * major[1] + v * minor[1]
    return ndimage.map_coordinates(mask.astype(np.uint8), [y, x], order=0, cval=0) > 0


def asymmetry_features(lab, mask):
    m = _mask_in_principal_frame(mask)
    area = m.sum()
    asym_major = (m ^ m[::-1, :]).sum() / area   # mirror across the major axis (v -> -v)
    asym_minor = (m ^ m[:, ::-1]).sum() / area   # mirror across the minor axis (u -> -u)
    # color asymmetry: mean Lab of the two halves on each side of each principal axis
    cy, cx, major, minor, ys, xs = _principal_frame(mask)
    u = (xs - cx) * major[0] + (ys - cy) * major[1]
    v = (xs - cx) * minor[0] + (ys - cy) * minor[1]
    px = lab[ys, xs]
    def half_delta(coord):
        a, b = px[coord >= 0], px[coord < 0]
        return float(np.linalg.norm(a.mean(0) - b.mean(0))) if len(a) and len(b) else 0.0
    return {"A_shape_asym_major": float(asym_major), "A_shape_asym_minor": float(asym_minor),
            "A_shape_asym_max": float(max(asym_major, asym_minor)),
            "A_color_asym_deltaE": max(half_delta(u), half_delta(v))}


def _box_count_dimension(boundary: np.ndarray) -> float:
    ys, xs = np.nonzero(boundary)
    if len(ys) < 10:
        return 1.0
    ys, xs = ys - ys.min(), xs - xs.min()
    extent = max(ys.max(), xs.max()) + 1
    sizes = [s for s in (2, 3, 4, 6, 8, 12, 16, 24, 32) if s < extent / 2]
    if len(sizes) < 3:
        return 1.0
    counts = [len(set(zip((ys // s).tolist(), (xs // s).tolist()))) for s in sizes]
    slope = np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)[0]
    return float(slope)


def border_features(lab, mask):
    area = mask.sum()
    perim = measure.perimeter(mask)
    hull = measure.regionprops(mask.astype(np.uint8))[0].image_convex
    hull_perim = measure.perimeter(hull)
    boundary = mask & ~ndimage.binary_erosion(mask)
    # radial irregularity: CV of centroid-to-boundary distance
    cy, cx = ndimage.center_of_mass(mask)
    by, bx = np.nonzero(boundary)
    radii = np.hypot(by - cy, bx - cx)
    # edge sharpness: L* gradient on the border, relative to the lesion/skin lightness step
    L = ndimage.gaussian_filter(lab[..., 0], 1.0)
    grad = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1)) / 8.0
    ring = ndimage.binary_dilation(mask, iterations=10) & ~mask
    step = abs(L[mask].mean() - L[ring].mean()) if ring.any() else 0.0
    g = grad[boundary]
    return {"B_circularity": float(4 * np.pi * area / perim ** 2) if perim else 0.0,
            "B_convexity": float(hull_perim / perim) if perim else 1.0,
            "B_fractal_dim": _box_count_dimension(boundary),
            "B_radial_cv": float(radii.std() / radii.mean()) if len(radii) else 0.0,
            "B_edge_sharpness": float(g.mean() / (step + 1.0)),
            "B_edge_sharpness_cv": float(g.std() / (g.mean() + 1e-6))}


_PROTO_LAB = color.rgb2lab(np.array(list(COLOR_PROTOTYPES_RGB.values()), dtype=float)[None] / 255.0)[0]


def color_features(lab, mask):
    px = lab[mask]
    ring = ndimage.binary_dilation(mask, iterations=15) & ~ndimage.binary_dilation(mask, iterations=5)
    skin = lab[ring].mean(0) if ring.any() else px.mean(0)
    protos = np.vstack([_PROTO_LAB, skin])                     # last prototype = this image's own skin
    nearest = np.argmin(((px[:, None, :] - protos[None]) ** 2).sum(-1), axis=1)
    fracs = np.bincount(nearest, minlength=len(protos)) / len(px)
    out = {f"C_frac_{name}": float(fracs[i]) for i, name in enumerate(COLOR_PROTOTYPES_RGB)}
    out["C_n_colors"] = int((fracs[:-1] >= COLOR_MIN_FRACTION).sum())
    ab_hist, _, _ = np.histogram2d(px[:, 1], px[:, 2], bins=16, range=[[-60, 80], [-40, 80]])
    p = ab_hist.ravel() / ab_hist.sum()
    p = p[p > 0]
    mean = px.mean(0)
    out.update({"C_color_entropy": float(-(p * np.log2(p)).sum()),
                "C_color_std": float(px.std(0).sum()),
                "C_deltaE_vs_skin": float(np.linalg.norm(mean - skin)),
                "C_darkness_vs_skin": float(skin[0] - mean[0]),
                "C_redness_vs_skin": float(mean[1] - skin[1]),
                "C_yellowness_vs_skin": float(mean[2] - skin[2])})
    return out


def texture_features(lab, mask):
    L = lab[..., 0]
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    Lc, mc = L[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    q = (np.clip(Lc, 0, 100) / 100 * 31).astype(np.uint8) + 1      # levels 1..32; 0 = outside lesion
    q[~mc] = 0
    glcm = feature.graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=33, symmetric=True, normed=False)[1:, 1:].astype(float)
    glcm /= glcm.sum(axis=(0, 1), keepdims=True) + 1e-12
    props = {k: float(feature.graycoprops(glcm, k).mean()) for k in ("contrast", "homogeneity", "energy", "correlation")}
    ent = -(glcm * np.log2(glcm + 1e-12)).sum(axis=(0, 1)).mean()
    lbp = feature.local_binary_pattern((np.clip(Lc, 0, 100) * 2.55).astype(np.uint8), P=8, R=1, method="uniform")
    hist = np.bincount(lbp[mc].astype(int), minlength=10) / mc.sum()
    h = hist[hist > 0]
    highpass = Lc - ndimage.gaussian_filter(Lc, 3)
    return {"T_glcm_contrast": props["contrast"], "T_glcm_homogeneity": props["homogeneity"],
            "T_glcm_energy": props["energy"], "T_glcm_correlation": props["correlation"],
            "T_glcm_entropy": float(ent), "T_lbp_entropy": float(-(h * np.log2(h)).sum()),
            "T_lbp_flat_fraction": float(hist[8]),   # uniform pattern 8 = all neighbours >= centre
            "T_roughness": float(highpass[mc].std())}


def shape_features(mask):
    props = measure.regionprops(mask.astype(np.uint8))[0]
    hu = props.moments_hu
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    out = {f"S_hu{i + 1}": float(v) for i, v in enumerate(hu)}
    out.update({"S_eccentricity": float(props.eccentricity), "S_solidity": float(props.solidity),
                "S_extent": float(props.extent),
                "S_axis_ratio": float(props.axis_minor_length / props.axis_major_length)
                if props.axis_major_length else 1.0})
    return out


def extract(img_rgb: np.ndarray, mask: np.ndarray) -> dict:
    """img_rgb: HxWx3 uint8; mask: HxW bool (single cleaned component). Returns {feature: value}."""
    if mask.sum() < 30:
        raise ValueError("mask too small")
    out = frame_features(mask)
    img, m = to_lesion_scale(shades_of_gray(img_rgb), mask)
    if m.sum() < 30:
        raise ValueError("mask too small after resampling")
    lab = color.rgb2lab(img / 255.0)
    out.update(asymmetry_features(lab, m))
    out.update(border_features(lab, m))
    out.update(color_features(lab, m))
    out.update(texture_features(lab, m))
    out.update(shape_features(m))
    return out


FEATURE_DICTIONARY = {
    "A_shape_asym_major": ("A", "0-2", "mask XOR its mirror across the major axis / area; 0 = symmetric"),
    "A_shape_asym_minor": ("A", "0-2", "same, mirrored across the minor axis"),
    "A_shape_asym_max": ("A", "0-2", "max of the two axes; ABCD scores asymmetry in 0, 1 or 2 axes"),
    "A_color_asym_deltaE": ("A", "deltaE76", "largest Lab difference between halves split on a principal axis"),
    "B_circularity": ("B", "0-1", "4*pi*area/perimeter^2; 1 = circle, lower = irregular border"),
    "B_convexity": ("B", "0-1", "hull perimeter / perimeter; lower = indented/notched border"),
    "B_fractal_dim": ("B", "1-2", "box-counting dimension of the border; higher = more ragged"),
    "B_radial_cv": ("B", "ratio", "CV of centroid-to-border distance; higher = irregular outline"),
    "B_edge_sharpness": ("B", "ratio", "border L* gradient / lesion-skin L* step; higher = abrupt edge"),
    "B_edge_sharpness_cv": ("B", "ratio", "variability of edge sharpness along the border"),
    "C_frac_white": ("C", "fraction", "share of lesion pixels nearest the white prototype"),
    "C_frac_red": ("C", "fraction", "... red prototype"),
    "C_frac_light_brown": ("C", "fraction", "... light-brown prototype"),
    "C_frac_dark_brown": ("C", "fraction", "... dark-brown prototype"),
    "C_frac_blue_gray": ("C", "fraction", "... blue-gray prototype"),
    "C_frac_black": ("C", "fraction", "... black prototype"),
    "C_n_colors": ("C", "count 0-6", "named colors covering >= 5% of the lesion (exploratory, unvalidated)"),
    "C_color_entropy": ("C", "bits", "entropy of the lesion a*b* histogram; higher = more color variety"),
    "C_color_std": ("C", "Lab units", "sum of L*, a*, b* standard deviations inside the lesion"),
    "C_deltaE_vs_skin": ("C", "deltaE76", "lesion mean color vs surrounding skin"),
    "C_darkness_vs_skin": ("C", "L* units", "skin L* minus lesion L*; higher = darker than skin"),
    "C_redness_vs_skin": ("C", "a* units", "lesion a* minus skin a*; higher = redder (erythema)"),
    "C_yellowness_vs_skin": ("C", "b* units", "lesion b* minus skin b*"),
    "T_glcm_contrast": ("T", "GLCM", "local L* contrast inside the lesion"),
    "T_glcm_homogeneity": ("T", "GLCM", "higher = smoother surface"),
    "T_glcm_energy": ("T", "GLCM", "higher = more uniform texture"),
    "T_glcm_correlation": ("T", "GLCM", "linear dependence of neighbouring L* values"),
    "T_glcm_entropy": ("T", "bits", "GLCM entropy; higher = more complex texture"),
    "T_lbp_entropy": ("T", "bits", "entropy of uniform LBP codes; higher = more varied micro-texture"),
    "T_lbp_flat_fraction": ("T", "fraction", "share of flat LBP patterns"),
    "T_roughness": ("T", "L* units", "std of high-pass L* inside the lesion (scale, crust, rough surface)"),
    "S_hu1": ("S", "-log10|hu|", "Hu moment 1 (spread)"),
    "S_hu2": ("S", "-log10|hu|", "Hu moment 2 (elongation)"),
    "S_hu3": ("S", "-log10|hu|", "Hu moment 3"),
    "S_hu4": ("S", "-log10|hu|", "Hu moment 4"),
    "S_hu5": ("S", "-log10|hu|", "Hu moment 5"),
    "S_hu6": ("S", "-log10|hu|", "Hu moment 6"),
    "S_hu7": ("S", "-log10|hu|", "Hu moment 7 (skew)"),
    "S_eccentricity": ("S", "0-1", "ellipse eccentricity; 0 = round"),
    "S_solidity": ("S", "0-1", "area / convex-hull area"),
    "S_extent": ("S", "0-1", "area / bounding-box area"),
    "S_axis_ratio": ("S", "0-1", "minor / major axis length"),
}
CONCEPT_GROUPS = {"A": "asymmetry", "B": "border", "C": "color", "T": "texture/surface", "S": "shape"}
# Not lesion features: kept in the table for QA, excluded from models and explanations.
FRAME_COLUMNS = {"frame_feret_rel": "max Feret diameter / image diagonal (depends on framing, not mm)",
                 "frame_area_rel": "lesion share of the image area (depends on framing)"}
