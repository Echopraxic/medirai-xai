"""
Download ISIC Archive images listed in a split/metadata CSV and store them resized like tools/cache_images.py
(longest side <= --max-side, Lanczos, JPEG q95), skipping files already present. Public CC images only; run where
there is internet (a cluster login node), then point training at --out.

    python tools/download_isic_resized.py --csv splits/isic_v3_clinical_plus_derm.csv --modality dermoscopic \
        --out $SCRATCH/medirai/datasets_768/ISIC_dermoscopic
"""
import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


def fetch_one(job):
    isic_id, url, out, max_side = job
    dst = Path(out) / f"{isic_id}.jpg"
    if dst.exists():
        return "skip"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with Image.open(io.BytesIO(r.content)) as im:
                im = im.convert("RGB")
                scale = max_side / max(im.size)
                if scale < 1:
                    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
                tmp = dst.with_suffix(".tmp")
                im.save(tmp, format="JPEG", quality=95)
                tmp.replace(dst)
            return "ok"
        except (requests.RequestException, OSError):
            time.sleep(2 ** attempt)
    return "fail"


def main(opts):
    df = pd.read_csv(opts.csv)
    if opts.modality:
        df = df[df.modality == opts.modality]
    df = df[df.split != "excluded"]
    if "url" not in df.columns or df.url.isna().any():
        meta = pd.read_csv(opts.urls)[["isic_id", "url"]]
        df = df.drop(columns=["url"], errors="ignore").merge(meta, on="isic_id", how="left")
    Path(opts.out).mkdir(parents=True, exist_ok=True)
    jobs = [(r.isic_id, r.url, opts.out, opts.max_side) for r in df.itertuples()]
    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(opts.threads) as ex:
        for n, status in enumerate(ex.map(fetch_one, jobs), 1):
            counts[status] += 1
            if n % 1000 == 0:
                print(n, len(jobs), counts, flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--urls", default="splits/sources/isic_dermoscopic_histo.csv", help="isic_id -> url, if --csv lacks urls")
    ap.add_argument("--modality", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--threads", type=int, default=16)
    main(ap.parse_args())
