"""
Predictions of a train_sprint run (e.g. the final model kdv8f_mel s3) on another image set, with exactly the inference
train_sprint.py uses for its own predictions.csv (single view, malignant if p >= 0.5). For the C1 surrogate's extra
rows (plan 2026-11-04); eval/export_predictions.py reads only uncertaintyNet configs.

    python eval/predict_sprint.py --run uncertaintyNet-main/output/narval_stageb_kdv8f_mel_s3_20261030/run \
        --split_csv splits/surrogate_extra_v1.csv --image_dir data/surrogate_extra/files_768 \
        --out eval/output/surrogate_c1/extra_predictions
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "classifier"), str(ROOT / "uncertaintyNet-main")]
import dataset  # noqa: E402
from train_sprint import load_for_inference, predict  # noqa: E402


def main(opts):
    out = Path(opts.out)
    if (out / "predictions.csv").resolve() == (Path(opts.run) / "predictions.csv").resolve():
        raise SystemExit("--out must not be the run folder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf = load_for_inference(opts.run, device)
    model.to(device)
    df = pd.read_csv(opts.split_csv)
    df = df[df.split != "excluded"].reset_index(drop=True)
    dl = DataLoader(dataset.ISICDatasetV2(df, opts.image_dir, tf), batch_size=opts.batch_size, num_workers=0)
    p, _, ids, y = predict(model, dl, device, device.type == "cuda")
    res = pd.DataFrame({"isic_id": ids, "label": y.astype(int), "pred": (p >= 0.5).astype(int), "p_malignant": p})
    if predict.last_mel is not None:
        res["p_melanoma"] = predict.last_mel
    out.mkdir(parents=True, exist_ok=True)
    res.merge(df[["isic_id", "diagnosis_3"]], on="isic_id").to_csv(out / "predictions.csv", index=False)
    print(f"{len(res)} images, malignant calls {res.pred.mean():.3f} -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    main(ap.parse_args())
