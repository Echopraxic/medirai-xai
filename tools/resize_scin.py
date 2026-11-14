"""
Resize every SCIN image to 768 px on the long side as <image_id>.jpg (image_id = file stem, as in splits/scin_v1.csv).
Images already present are skipped; unreadable files are listed. Run on Narval next to the SCIN download.

    python tools/resize_scin.py --scin_dir /scratch/mikeymck/medirai/external/scin --out <scin_dir>/files_768
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main(opts):
    d = Path(opts.scin_dir)
    cases = pd.read_csv(d / "scin_cases.csv", dtype=str)
    names = {Path(p).name for k in (1, 2, 3) for p in cases[f"image_{k}_path"].dropna()}
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = []
    for name in sorted(names):
        dst = out / f"{Path(name).stem}.jpg"
        if dst.exists():
            continue
        try:
            im = Image.open(d / "images" / name).convert("RGB")
            im.thumbnail((768, 768))
            im.save(dst, quality=92)
        except (OSError, ValueError):
            bad.append(name)
    print(f"{len(names)} image files, {len(bad)} unreadable: {bad[:20]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scin_dir", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
