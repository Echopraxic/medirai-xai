"""
Resize the extracted PASSION images to 768 px on the long side as <image_id>.jpg (image_id = file stem, e.g. AA00970008_1).
PASSION photos are up to ~100 MP, so PIL's decompression-bomb limit is lifted for these known files. EXIF is not copied.
Images already present are skipped; unreadable files are listed. Local only (plan 2026-11-04: PASSION never leaves the laptop).

    python tools/resize_passion.py --images data/passion/PASSION_MICCAI_2024/images --out data/passion/files_768
"""
import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main(opts):
    src, out = Path(opts.images), Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    bad = []
    for p in files:
        dst = out / f"{p.stem}.jpg"
        if dst.exists():
            continue
        try:
            im = Image.open(p)
            im.draft("RGB", (1536, 1536))  # fast JPEG downscale on decode; thumbnail below sets the final size
            im = im.convert("RGB")
            im.thumbnail((768, 768))
            im.save(dst, quality=92)
        except (OSError, ValueError):
            bad.append(p.name)
    print(f"{len(files)} image files, {len(bad)} unreadable: {bad[:20]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
