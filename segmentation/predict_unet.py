"""
Predict lesion masks for the ISIC clinical close-ups with the trained ResNet18-UNet (flip TTA).

Writes <out>/<isic_id>.png (0/255, at the cached image's resolution) and <out>/unet_stats.csv with
per-image area ratio, components before cleanup, border touch, solidity and boundary uncertainty
(share of mask-area pixels whose probability is in 0.3-0.7).

    python segmentation/predict_unet.py --ckpt segmentation/output/unet_r18/best.pt \
        --images uncertaintyNet-main/datasets_768/ISIC_clinical \
        --out uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from postprocess import border_touch_fraction, clean_mask, solidity
from unet_resnet import IMAGENET_MEAN, IMAGENET_STD, ResNetUNet

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predict_prob(net, img: Image.Image, size: int, device) -> np.ndarray:
    tf = v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True),
                     v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    x = tf(img)[None].to(device)
    batch = torch.cat([x, x.flip(-1), x.flip(-2)])
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.sigmoid(net(batch).float())
    p = (p[0] + p[1].flip(-1) + p[2].flip(-2)) / 3
    p = F.interpolate(p[None], size=(img.height, img.width), mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy()


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(opts.ckpt, map_location=device)
    net = ResNetUNet(pretrained=False).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(ROOT / "splits" / opts.split)
    ids = split.loc[split.split != "excluded", "isic_id"].tolist()
    rows = []
    for n, isic_id in enumerate(ids, 1):
        img = Image.open(Path(opts.images) / f"{isic_id}.jpg").convert("RGB")
        prob = predict_prob(net, img, ckpt["size"], device)
        mask, n_comp = clean_mask(prob > 0.5)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{isic_id}.png")
        area = int(mask.sum())
        uncertain = ((prob > 0.3) & (prob < 0.7)).sum()
        rows.append({"isic_id": isic_id, "unet_area_ratio": area / mask.size, "unet_components": n_comp,
                     "unet_border_touch": border_touch_fraction(mask), "unet_solidity": solidity(mask),
                     "unet_boundary_uncertainty": float(uncertain / max(area, 1)),
                     "unet_mean_prob_in_mask": float(prob[mask].mean()) if area else 0.0})
        if n % 1000 == 0:
            print(f"{n}/{len(ids)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "unet_stats.csv", index=False)
    print(f"done: {len(rows)} masks -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="isic_clinical_v2.csv", help="file name under splits/")
    main(ap.parse_args())
