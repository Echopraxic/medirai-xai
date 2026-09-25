"""
Download metadata (no images) for every ISIC Archive image matching a query,
flattened to one row per image. Used to build the persisted train/val/test split.

Usage:
    python splits/fetch_isic_metadata.py \
        --query 'image_type:"clinical: close-up"' \
        --out splits/isic_clinical_closeup_metadata.csv
"""
import argparse
import time

import pandas as pd
import requests

API = "https://api.isic-archive.com/api/v2/images/search/"


def flatten(record: dict) -> dict:
    row = {
        "isic_id": record["isic_id"],
        "copyright_license": record.get("copyright_license"),
        "attribution": record.get("attribution"),
    }
    for section, values in (record.get("metadata") or {}).items():
        for key, value in values.items():
            row[key] = value  # section keys (acquisition, clinical, patient...) don't collide
    return row


def fetch(query: str, page_size: int = 100, pause_s: float = 0.2) -> pd.DataFrame:
    rows = []
    url, params = API, {"query": query, "limit": page_size}
    while url:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows.extend(flatten(r) for r in payload["results"])
        print(f"\r{len(rows)}/{payload['count']}", end="", flush=True)
        url, params = payload.get("next"), None  # 'next' already encodes the cursor and query
        time.sleep(pause_s)
    print()
    return pd.DataFrame(rows).sort_values("isic_id").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default='image_type:"clinical: close-up"')
    parser.add_argument("--out", default="splits/isic_clinical_closeup_metadata.csv")
    opts = parser.parse_args()

    df = fetch(opts.query)
    df.to_csv(opts.out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {opts.out}")
