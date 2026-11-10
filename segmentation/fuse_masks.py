"""
Combine two mask sets (e.g. clinical UNet and box-prompted SAM) into a third, for segmentation/gold_eval.py to score.
No training; rules are chosen on the gold `tune` half only.

Rules (a = --a masks, b = --b masks; both 0/255 PNGs named <isic_id>.png at the cached image size):
  and     a & b                              (removes leaks only one method makes)
  or      a | b                              (recovers lesion parts only one method finds)
  agree   a & b if Dice(a, b) >= --min_agreement else a  (trust the overlap only when the two broadly agree)
Each fused mask is cleaned like the originals (largest component, holes filled). An image missing from b keeps a.

    python segmentation/fuse_masks.py --a uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical \
        --b uncertaintyNet-main/datasets_768/ISIC_clinical_masks/sam_box --rule and \
        --out uncertaintyNet-main/datasets_768/ISIC_clinical_masks/fuse_and
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from postprocess import clean_mask, dice


def fuse(a: np.ndarray, b: np.ndarray, rule: str, min_agreement: float = 0.6) -> np.ndarray:
    if rule == "and":
        out = a & b
    elif rule == "or":
        out = a | b
    elif rule == "agree":
        out = a & b if dice(a, b) >= min_agreement else a
    else:
        raise ValueError(rule)
    if not out.any():  # an empty overlap would delete the lesion: keep a
        out = a
    return clean_mask(out)[0]


def main(opts):
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for pa in sorted(Path(opts.a).glob("*.png")):
        a = np.array(Image.open(pa)) > 127
        pb = Path(opts.b) / pa.name
        if pb.exists():
            b = np.array(Image.open(pb)) > 127
            if b.shape != a.shape:
                b = np.array(Image.fromarray(b.astype(np.uint8) * 255).resize(a.shape[::-1], Image.NEAREST)) > 127
            m = fuse(a, b, opts.rule, opts.min_agreement)
        else:
            m = a
        Image.fromarray((m * 255).astype(np.uint8)).save(out / pa.name)
        n += 1
    print(f"done: {n} masks -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--rule", required=True, choices=["and", "or", "agree"])
    ap.add_argument("--min_agreement", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
