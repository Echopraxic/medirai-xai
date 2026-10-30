"""
Handcrafted v1.0 features on external images: the clinical fine-tuned UNet mask, then features/extract.py, exactly as
the feature table v1.0 but for images outside split v2. For S1-9 (TreeSHAP on the final model's locked-set errors,
descriptive only) and the darkness-contrast check on the darker-skin dev set (decision 2026-10-30).

Sources: dsd (splits/f17k_dsd_v1.csv, selection data; images in data/f17k_dsd/files_768) and the four locked sets
(ddi, ddi2, mcr_sl, isic_overview, built by eval/external_test.py). No model is scored here and nothing is selected,
but a locked set's pixels are read again, so a locked source needs --descriptive '<reason>' and is logged in
eval/output/external/looks.csv as a descriptive look (not a --final look). Outputs fall under each source's terms:
DDI/DDI-2 rows are Stanford AIMI data (research use, keep local).

    python features/extract_external.py --source dsd
    python features/extract_external.py --source ddi --descriptive 'S1-9 TreeSHAP of final-model errors'
        -> features/output/external/<source>_features_v1.0.parquet
"""
import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "features"), str(ROOT / "segmentation"), str(ROOT / "eval")]
import extract as fx  # noqa: E402
from external_test import LOCKED, LOOK_LOG, SOURCES, load_768  # noqa: E402
from postprocess import border_touch_fraction, clean_mask  # noqa: E402
from predict_unet import predict_prob  # noqa: E402
from unet_resnet import ResNetUNet  # noqa: E402

CKPT = ROOT / "segmentation/output/unet_r18_clinical/best.pt"


def build_dsd():
    d = pd.read_csv(ROOT / "splits/f17k_dsd_v1.csv")
    return pd.DataFrame({"image_id": d.isic_id, "path": [str(ROOT / "data/f17k_dsd/files_768" / f"{i}.jpg") for i in d.isic_id],
                         "label": d.label.astype(int), "group": d.group_id, "diagnosis": d.diagnosis_3,
                         "fitzpatrick": d.fitzpatrick, "dsd": d.dsd})


def log_descriptive(source, reason):
    row = pd.DataFrame([{"time": datetime.datetime.now().isoformat(timespec="seconds"), "source": source,
                         "runs": "features v1.0 (descriptive, no model scored)", "reason": f"DESCRIPTIVE: {reason}"}])
    row.to_csv(LOOK_LOG, mode="a", header=not LOOK_LOG.exists(), index=False)


def main(opts):
    if opts.source in LOCKED:
        if not opts.descriptive:
            raise SystemExit(f"{opts.source} is locked: pass --descriptive '<reason>' (logged to looks.csv)")
        log_descriptive(opts.source, opts.descriptive)
    df = (build_dsd() if opts.source == "dsd" else SOURCES[opts.source]()).reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device)
    net = ResNetUNet(pretrained=False).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    rows = []
    for n, (i, path) in enumerate(zip(df.image_id, df.path), 1):
        img = load_768(path)
        prob = predict_prob(net, img, ckpt["size"], device)
        mask, _ = clean_mask(prob > 0.5)
        area = int(mask.sum())
        qa = {"unet_area_ratio": area / mask.size, "unet_border_touch": border_touch_fraction(mask),
              "unet_mean_prob_in_mask": float(prob[mask].mean()) if area else 0.0}
        try:
            rows.append({"image_id": i, **qa, **fx.extract(np.array(img), mask), "extraction_error": ""})
        except Exception as err:  # keep the row; the error is reported
            rows.append({"image_id": i, **qa, "extraction_error": f"{type(err).__name__}: {err}"})
        if n % 200 == 0:
            print(f"{n}/{len(df)}", flush=True)
    out = ROOT / "features/output/external" / f"{opts.source}_features_{fx.FEATURE_VERSION}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = df.drop(columns="path").merge(pd.DataFrame(rows), on="image_id", how="left")
    table.insert(1, "feature_version", fx.FEATURE_VERSION)
    table.to_parquet(out, index=False)
    print(f"{opts.source}: {len(table)} images, {int((table.extraction_error != '').sum())} extraction errors -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["dsd", *sorted(SOURCES)])
    ap.add_argument("--descriptive", help="reason, required for a locked source (logged to looks.csv)")
    main(ap.parse_args())
