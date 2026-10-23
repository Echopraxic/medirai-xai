"""
Download a public Mendeley Data dataset (all files of one version) over the public API, SHA-256 checked and resumable.

    python tools/download_mendeley.py bb844rcjx5 --version 1 --out /scratch/mikeymck/medirai/external/mendeley_synthetic_fst
"""
import argparse
import hashlib
import json
from pathlib import Path

import requests

API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={v}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(opts):
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    files = requests.get(API.format(id=opts.dataset, v=opts.version), timeout=60).json()
    status = {}
    for f in files:
        dest, want = out / f["filename"], f["content_details"]["sha256_hash"]
        if dest.exists() and sha256(dest) == want:
            status[f["filename"]] = "skipped (verified)"
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(f["content_details"]["download_url"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        ok = sha256(tmp) == want
        if ok:
            tmp.replace(dest)
        status[f["filename"]] = "ok" if ok else "sha256 MISMATCH"
        print(f["filename"], status[f["filename"]], flush=True)
    (out / "manifest.json").write_text(json.dumps({"dataset": opts.dataset, "version": opts.version, "files": status}, indent=1))
    if any("MISMATCH" in v for v in status.values()):
        raise SystemExit("some files failed the sha256 check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
