"""Mask clean-up and quality measures shared by the UNet and SAM mask scripts."""
import numpy as np
from scipy import ndimage
from skimage import measure


def clean_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Largest connected component with holes filled. Returns (mask, number of components before cleanup).

    No convex hull here: the hull would erase the border irregularity the B features measure. Solidity
    (area / hull area) is reported as a quality signal instead.
    """
    labels, n = ndimage.label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(largest), n


def border_touch_fraction(mask: np.ndarray) -> float:
    """Fraction of the image border covered by the mask (lesion cut off by the frame, or mask leaking)."""
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(edge.mean())


def solidity(mask: np.ndarray) -> float:
    props = measure.regionprops(mask.astype(np.uint8))
    return float(props[0].solidity) if props else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    denom = a.sum() + b.sum()
    return float(2 * (a & b).sum() / denom) if denom else 1.0


def quality_flag(area_ratio: float, border_touch: float, agreement: float | None) -> str:
    """'ok' / 'review' / 'fail'. Thresholds are deliberately simple and reported with every feature table."""
    if area_ratio < 0.002 or area_ratio > 0.85:
        return "fail"
    if agreement is not None and agreement < 0.5:
        return "fail"
    if border_touch > 0.25 or (agreement is not None and agreement < 0.75):
        return "review"
    return "ok"


def quality_flag_v1(area_ratio: float, border_touch: float, agreement: float, mean_prob: float) -> str:
    """Flag for the clinical fine-tuned UNet masks (feature table v1). 'ok' / 'review' / 'fail'.

    Thresholds were fit on the `tune` half of the hand-annotated gold set (segmentation/gold, bad = Dice < 0.8 or
    no visible lesion) and checked on the `report` half: bad-mask rate ok 13% / review 67% / fail 100% (tune),
    ok 19% / review 64% / fail 56% (report). The UNet's mean probability inside its own mask is the strongest
    single signal (gold AUROC 0.91 / 0.88), agreement with SAM next (0.89 / 0.79).
    """
    if area_ratio < 0.002 or area_ratio > 0.85 or mean_prob < 0.80 or agreement < 0.4:
        return "fail"
    if mean_prob >= 0.85 and agreement >= 0.6 and area_ratio <= 0.3 and border_touch <= 0.25:
        return "ok"
    return "review"
