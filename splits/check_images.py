"""
Check that every image in the persisted split is present and readable, and optionally download
missing or corrupt ones from the public ISIC Archive. (Student 1 handoff, Step 3.)

Usage (on the VM):
    python splits/check_images.py --data-dir datasets/ISIC_clinical              # report only
    python splits/check_images.py --data-dir datasets/ISIC_clinical --download   # fetch what's missing

Writes <data-dir>/image_check.csv listing every problem image. Exit code 0 only when nothing is missing.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE_API = "https://api.isic-archive.com/api/v2/images/{isic_id}/"


def image_status(path: Path) -> str:
    """'ok', 'missing' or 'corrupt' (unreadable or truncated file)."""
    if not path.exists():
        return "missing"
    try:
        with Image.open(path) as img:
            img.load()  # decodes the full image, so truncated downloads are caught
        return "ok"
    except Exception:
        return "corrupt"


def check(split_file: Path, data_dir: Path, include_excluded: bool = False) -> pd.DataFrame:
    split = pd.read_csv(split_file)
    if not include_excluded:
        split = split[split["split"] != "excluded"]
    rows = [{"isic_id": i, "split": s, "status": image_status(data_dir / f"{i}.jpg")}
            for i, s in zip(split["isic_id"], split["split"])]
    return pd.DataFrame(rows)


def download(isic_id: str, data_dir: Path, session=requests, retries: int = 3) -> str:
    """Download one full-resolution image via the ISIC API; returns the resulting status."""
    for attempt in range(retries):
        try:
            meta = session.get(IMAGE_API.format(isic_id=isic_id), timeout=60)
            meta.raise_for_status()
            full = meta.json()["files"]["full"]
            response = session.get(full["url"], timeout=120)
            response.raise_for_status()
            # Compare with the HTTP Content-Length, not the API's "size": S3 serves the JPEG with ~3 KB of
            # embedded XMP licence metadata that the API size does not count. image_status() then fully
            # decodes the file, which catches any remaining truncation.
            expected = response.headers.get("Content-Length")
            if expected is not None and len(response.content) != int(expected):
                raise IOError(f"truncated download ({len(response.content)} of {expected} bytes)")
            target = data_dir / f"{isic_id}.jpg"
            tmp = target.with_suffix(".part")
            tmp.write_bytes(response.content)
            tmp.replace(target)  # never leave a half-written .jpg behind
            return image_status(target)
        except Exception as err:  # network hiccup: back off and retry
            last = err
            time.sleep(2 ** attempt)
    print(f"\n  failed {isic_id}: {last}", file=sys.stderr)
    return "download_failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder holding <isic_id>.jpg files")
    parser.add_argument("--split-file", default=str(HERE / "isic_clinical_v1.csv"))
    parser.add_argument("--download", action="store_true", help="download missing/corrupt images")
    parser.add_argument("--include-excluded", action="store_true",
                        help="also check the 1,333 excluded (Indeterminate/unlabelled) images")
    opts = parser.parse_args()

    data_dir = Path(opts.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    report = check(Path(opts.split_file), data_dir, opts.include_excluded)
    problems = report[report["status"] != "ok"]
    print(f"{len(report) - len(problems)}/{len(report)} images ok; "
          f"{(problems['status'] == 'missing').sum()} missing, {(problems['status'] == 'corrupt').sum()} corrupt")

    if opts.download and len(problems):
        session = requests.Session()
        for n, isic_id in enumerate(problems["isic_id"], 1):
            report.loc[report["isic_id"] == isic_id, "status"] = download(isic_id, data_dir, session)
            print(f"\rdownloaded {n}/{len(problems)}", end="", flush=True)
            time.sleep(0.1)
        print()
        problems = report[report["status"] != "ok"]

    out = data_dir / "image_check.csv"
    problems.to_csv(out, index=False)
    print(f"{len(problems)} problem images listed in {out}")
    sys.exit(0 if problems.empty else 1)
