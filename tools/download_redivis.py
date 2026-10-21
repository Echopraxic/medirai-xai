"""
Download Redivis datasets (metadata tables + file-index tables) over the REST API, md5-checked and resumable.

The redivis Python client needs niquests/polars/geopandas, which don't install from the Alliance wheelhouse, so this
uses `requests` only. The API token is read from $REDIVIS_API_TOKEN or ~/.redivis_token and never printed.

    python tools/download_redivis.py --out /scratch/mikeymck/medirai/external \
        aimi.mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer:dpx2:v1_0 \
        aimi.ddi_diverse_dermatology_images:3r16:v1_0 aimi.ddi2_diverse_dermatology_images_2:6f7f:v1_0

Output per dataset (folder = the dataset name without owner/version): <table>.csv for every table (file-index
tables included, as the file list) and files/<file_name> for every file; manifest.json with row counts and md5 status.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://redivis.com/api/v1"


def token():
    tok = os.environ.get("REDIVIS_API_TOKEN") or Path("~/.redivis_token").expanduser().read_text().strip()
    return {"Authorization": f"Bearer {tok}"}


def get(path, headers, **params):
    for attempt in range(5):
        r = requests.get(f"{API}/{path}", headers=headers, params=params, timeout=120)
        if r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(2 ** attempt)
    r.raise_for_status()


def table_rows(ref, headers, n_rows):
    rows = [json.loads(line) for line in get(f"tables/{ref}/rows", headers, maxResults=n_rows + 10).text.splitlines()
            if line.strip()]
    if len(rows) != n_rows:
        raise RuntimeError(f"{ref}: got {len(rows)} rows, table has {n_rows}")
    return pd.DataFrame(rows)


def md5_b64(path):
    return base64.b64encode(hashlib.md5(Path(path).read_bytes()).digest()).decode()


def fetch_file(row, folder, headers):
    dest = folder / row["file_name"]
    if dest.exists() and md5_b64(dest) == row["md5_hash"]:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)  # some releases keep files in subfolders (MIDAS: .ipynb_checkpoints/)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with requests.get(f"{API}/rawFiles/{row['file_id']}", headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if md5_b64(tmp) == row["md5_hash"]:
                tmp.replace(dest)
                return "ok"
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return "failed"


def download(ref, out, headers, workers):
    name = ref.split(".", 1)[1].split(":")[0]
    folder = Path(out) / name
    (folder / "files").mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": ref, "tables": {}}
    for t in get(f"datasets/{ref}/tables", headers, maxResults=100).json()["results"]:
        tname = t["qualifiedReference"].rsplit(".", 1)[1].split(":")[0]
        df = table_rows(t["qualifiedReference"], headers, int(t["numRows"]))
        df.to_csv(folder / f"{tname}.csv", index=False)
        entry = {"rows": len(df), "file_index": bool(t.get("isFileIndex"))}
        if entry["file_index"]:
            if df.file_name.duplicated().any():
                raise RuntimeError(f"{tname}: duplicate file names; files/ would collide")
            with ThreadPoolExecutor(workers) as ex:
                status = list(ex.map(lambda r: fetch_file(r, folder / "files", headers), df.to_dict("records")))
            entry["files"] = {s: status.count(s) for s in set(status)}
        manifest["tables"][tname] = entry
        print(name, tname, entry, flush=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", help="qualified references, e.g. aimi.ddi_diverse_dermatology_images:3r16:v1_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    opts = ap.parse_args()
    h = token()
    failed = [m["dataset"] for m in (download(d, opts.out, h, opts.workers) for d in opts.datasets)
              if any(e.get("files", {}).get("failed") for e in m["tables"].values())]
    sys.exit(f"files failed md5/download in: {failed}" if failed else 0)
