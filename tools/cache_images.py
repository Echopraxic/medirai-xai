"""
Build resized image caches so training/inference does not decode full-resolution JPEGs every epoch.

Clinical close-ups range up to 6000x4000; the models resize to 224-384 anyway. Images are resized so the
longest side is at most --max-side (aspect ratio kept, Lanczos), masks with nearest-neighbour.
Downstream transforms (e.g. Resize((224, 224)) in uncertaintyNet's MEDIRV2 loader) are unchanged; the
only difference from reading originals is one extra antialiased downsampling step.

    # ISIC clinical close-ups -> <out>/ISIC_clinical/<isic_id>.jpg
    python tools/cache_images.py clinical --src uncertaintyNet-main/datasets/ISIC_clinical \
        --out uncertaintyNet-main/datasets_768

    # ISIC 2018 Task 1 zips (already unpacked) -> <out>/ISIC2018/{train,val,test}/{img,seg}
    python tools/cache_images.py isic2018 --src uncertaintyNet-main/datasets/ISIC2018_raw \
        --out uncertaintyNet-main/datasets_768
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ISIC2018_PARTS = {
    "train": ("ISIC2018_Task1-2_Training_Input", "ISIC2018_Task1_Training_GroundTruth"),
    "val": ("ISIC2018_Task1-2_Validation_Input", "ISIC2018_Task1_Validation_GroundTruth"),
    "test": ("ISIC2018_Task1-2_Test_Input", "ISIC2018_Task1_Test_GroundTruth"),
}


def resize_one(job):
    src, dst, max_side, is_mask = job
    dst = Path(dst)
    if dst.exists():
        return
    with Image.open(src) as im:
        im = im.convert("L" if is_mask else "RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(size, Image.NEAREST if is_mask else Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
        if is_mask:
            im.save(tmp, format="PNG")
        else:
            im.save(tmp, format="JPEG", quality=95)
        tmp.replace(dst)


def clinical_jobs(src, out, max_side):
    return [(p, out / "ISIC_clinical" / p.name, max_side, False) for p in sorted(src.glob("*.jpg"))]


def isic2018_jobs(src, out, max_side):
    jobs = []
    for part, (img_dir, seg_dir) in ISIC2018_PARTS.items():
        for p in sorted((src / img_dir).glob("*.jpg")):
            jobs.append((p, out / "ISIC2018" / part / "img" / p.name, max_side, False))
        for p in sorted((src / seg_dir).glob("*_segmentation.png")):
            jobs.append((p, out / "ISIC2018" / part / "seg" / p.name, max_side, True))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["clinical", "isic2018"])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--workers", type=int, default=6)
    opts = ap.parse_args()

    make = clinical_jobs if opts.kind == "clinical" else isic2018_jobs
    jobs = make(opts.src, opts.out, opts.max_side)
    with ProcessPoolExecutor(opts.workers) as ex:
        for n, _ in enumerate(ex.map(resize_one, jobs, chunksize=16), 1):
            if n % 1000 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"done: {len(jobs)} files -> {opts.out}")
