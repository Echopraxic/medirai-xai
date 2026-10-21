"""
Stage A1 of the classifier sprint: test-time preprocessing on existing checkpoints, scored on the SELECTION data only
(split v2 val + MIDAS dev_val phone photos; pre-registration 2026-10-21). No locked or test split is read.

Variants (all end in the run's MEDIRV2 test transform at its input size):
  stretch     current: whole image resized to a square (aspect ratio distorted)
  pad         whole image, aspect ratio kept, padded to a square with the image's median border colour
  center_50 / center_33 / center_25   square centre crop of 1/2, 1/3, 1/4 of the short side (MIDAS photos are aimed
              at the lesion but framed wide: 6 in / 1 ft)
  unet        square crop around the clinical UNet mask's bounding box (2x margin); full image if the mask is empty
  <best>+tta  8 dihedral views (flips x 90-degree rotations), softmax averaged

Score S = mean(internal val AUROC, MIDAS dev_val AUROC), reported per checkpoint and averaged per recipe.

    python eval/preprocess_ablation.py --runs rorqual_v2_lr3e-4_20261014/resnet50_det/run ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))

import dataset  # noqa: E402
import utils  # noqa: E402
from export_predictions import build_network, load_best  # noqa: E402

VARIANTS = ["stretch", "pad", "center_50", "center_33", "center_25", "unet"]


def selection_sets():
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    v2 = v2[v2.split == "val"]
    internal = pd.DataFrame({"image_id": v2.isic_id, "label": v2.label.astype(int),
                             "path": [str(UNC / "datasets_768/ISIC_clinical" / f"{i}.jpg") for i in v2.isic_id],
                             "mask": [str(UNC / "datasets_768/ISIC_clinical_masks/unet_clinical" / f"{i}.png") for i in v2.isic_id]})
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    m = m[(m.split == "dev_val") & m.modality.str.startswith("phone")]
    midas = pd.DataFrame({"image_id": m.image_id, "label": m.label.astype(int),
                          "path": [str(ROOT / "data/midas/files_768" / f"{i}.jpg") for i in m.image_id],
                          "mask": [str(ROOT / "data/midas/masks_unet_clinical" / f"{i}.png") for i in m.image_id]})
    return {"internal_val": internal.reset_index(drop=True), "midas_dev_val": midas.reset_index(drop=True)}


def square_crop(im, cx, cy, side):
    half = side / 2
    return im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))  # PIL pads outside with black


def apply(variant, im, mask_path):
    w, h = im.size
    if variant == "stretch":
        return im
    if variant == "pad":
        border = np.concatenate([np.asarray(im)[0], np.asarray(im)[-1], np.asarray(im)[:, 0], np.asarray(im)[:, -1]])
        return ImageOps.pad(im, (max(w, h), max(w, h)), color=tuple(int(v) for v in np.median(border, 0)))
    if variant.startswith("center_"):
        side = min(w, h) * int(variant.split("_")[1]) / 100
        return square_crop(im, w / 2, h / 2, side)
    if variant == "unet":
        mk = np.asarray(Image.open(mask_path).resize((w, h), Image.NEAREST)) > 127
        if not mk.any():
            return im
        ys, xs = np.nonzero(mk)
        side = max(xs.max() - xs.min(), ys.max() - ys.min(), 16) * 2
        return square_crop(im, (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, min(side, max(w, h)))
    raise ValueError(variant)


DIHEDRAL = [lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2),
            lambda x: x.transpose(-1, -2), lambda x: x.transpose(-1, -2).flip(-1),
            lambda x: x.transpose(-1, -2).flip(-2), lambda x: x.transpose(-1, -2).flip(-1).flip(-2)]


@torch.no_grad()
def predict(net, tf, images, device, tta=False, batch_size=32):
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        views = DIHEDRAL if tta else DIHEDRAL[:1]
        p = torch.stack([F.softmax(net(v(x)).float(), -1) for v in views]).mean(0)
        out.append(p[:, 1].cpu().numpy())
    return np.concatenate(out)


def main(opts):
    sets = selection_sets()
    raw = {name: [Image.open(p).convert("RGB") for p in d.path] for name, d in sets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run in opts.runs:
        folder = UNC / "output" / run
        config = utils.read_config(folder)
        net = build_network(config, folder)
        load_best(net, folder)
        net.to(device).eval()
        _, tf = dataset.medirv2_transforms(config["Network"]["Basic Setup"]["input_size"])
        for variant in VARIANTS + [f"{v}+tta" for v in opts.tta_on]:
            base = variant.split("+")[0]
            row = {"run": run.split("/")[0], "variant": variant}
            for name, d in sets.items():
                ims = [apply(base, im, mp) for im, mp in zip(raw[name], d["mask"])]
                p = predict(net, tf, ims, device, tta=variant.endswith("+tta"))
                row[f"{name}_auroc"] = float(roc_auc_score(d.label, p))
            row["S"] = (row["internal_val_auroc"] + row["midas_dev_val_auroc"]) / 2
            rows.append(row)
            print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    df["recipe"] = df.run.str.replace(r"_s1_|_seed1_", "_", regex=True).str.replace("narval_v2_lr", "v2_lr")
    df["recipe"] = np.where(df.run.str.contains("medir_v3"), "v3", "v2")
    out = ROOT / "eval/output/stage_a/preprocess_ablation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_checkpoint.csv", index=False)
    agg = df.groupby(["recipe", "variant"])[["internal_val_auroc", "midas_dev_val_auroc", "S"]].mean().round(4)
    agg = agg.sort_values("S", ascending=False)
    (out / "summary.md").write_text("# Stage A1: test-time preprocessing (selection data only)\n\n"
                                    "Mean over the checkpoints of each recipe (v2: seeds 0, 1; v3: seeds 0, 1).\n\n"
                                    + agg.to_markdown() + "\n", encoding="utf-8")
    print(agg.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta_on", nargs="*", default=["stretch", "center_50", "center_33"])
    main(ap.parse_args())
