"""
Fetch ISIC Archive metadata (no images) for a candidate data source, keyed by isic_id, for split v3 planning (WS9).

Default query: biopsy-confirmed dermoscopic images. Writes one CSV with diagnosis levels 1-3, lesion_id, patient_id,
licence, attribution and the full-image URL, then reports overlap with the current split by lesion_id/patient_id
(a dermoscopic image of a lesion already in our val/test must never be used for training).

    python splits/fetch_isic_metadata.py --out splits/sources/isic_dermoscopic_histo.csv
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.isic-archive.com/api/v2/images/search/"
DEFAULT_QUERY = 'image_type:"dermoscopic" AND diagnosis_confirm_type:"histopathology"'


def fetch(query, page_size=100):
    rows, url, params = [], API, {"query": query, "limit": page_size}
    while url:
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"ISIC API failed at {url}")
        page = r.json()
        for im in page["results"]:
            clin = im.get("metadata", {}).get("clinical", {})
            acq = im.get("metadata", {}).get("acquisition", {})
            rows.append({"isic_id": im["isic_id"], "image_type": acq.get("image_type"),
                         "diagnosis_1": clin.get("diagnosis_1"), "diagnosis_2": clin.get("diagnosis_2"),
                         "diagnosis_3": clin.get("diagnosis_3"), "diagnosis_confirm_type": clin.get("diagnosis_confirm_type"),
                         "lesion_id": clin.get("lesion_id"), "patient_id": clin.get("patient_id"),
                         "anatom_site_general": clin.get("anatom_site_general"), "fitzpatrick_skin_type": clin.get("fitzpatrick_skin_type"),
                         "attribution": im.get("attribution"), "copyright_license": im.get("copyright_license"),
                         "url": im.get("files", {}).get("full", {}).get("url")})
        url, params = page.get("next"), None
        if len(rows) % 5000 < page_size:
            print(f"{len(rows)} / {page.get('count')}", flush=True)
    return pd.DataFrame(rows)


def main(opts):
    df = fetch(opts.query)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    split = pd.read_csv(ROOT / "splits" / "isic_clinical_v2.csv")
    print(f"{len(df)} images, {df.lesion_id.nunique()} lesions, {df.patient_id.nunique()} patients -> {out}")
    print("diagnosis_1:", df.diagnosis_1.value_counts().to_dict())
    print("licence:", df.copyright_license.value_counts().to_dict())
    for key in ["lesion_id", "patient_id"]:
        ours = split.dropna(subset=[key])
        hit = df[df[key].isin(ours[key])]
        by_split = ours.set_index(key).split
        by_split = by_split[~by_split.index.duplicated()]
        print(f"shared {key}: {hit[key].nunique()} ids / {len(hit)} images; our split of those ids:",
              hit[key].map(by_split).value_counts().to_dict())
    print("attribution (top 10):", df.attribution.value_counts().head(10).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--out", default=str(ROOT / "splits/sources/isic_dermoscopic_histo.csv"))
    main(ap.parse_args())
