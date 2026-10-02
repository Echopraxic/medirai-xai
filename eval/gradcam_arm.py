"""
Grad-CAM comparison arm for S1's ResNet50 (uncertaintyNet): CAM for the model's own predicted class.

Per test image:
  cam_in_lesion      share of CAM mass inside the lesion mask (plausibility; mask from segmentation/)
  pointing_hit       CAM argmax falls inside the (slightly dilated) lesion
  deletion_auc       mean p(predicted class) while the most-salient pixels are progressively blurred out
                     (lower = more faithful); random_deletion_auc is the same with a random pixel order
  insertion_auc      mean p(predicted class) while salient pixels are restored onto a blurred image (higher = better)
Writes gradcam_metrics.csv and cams.npz (224x224 float16 CAMs) to --out; overlays for --overlay-ids.

    python eval/gradcam_arm.py --run s2_local/resnet50_det/run_X --dataset-path uncertaintyNet-main/datasets_768 \
        --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet --out eval/output/gradcam
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy import ndimage
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_predictions import UNC, build_network, load_best  # noqa: E402

import dataset  # noqa: E402  (uncertaintyNet-main, path set by export_predictions)
import utils  # noqa: E402

STEPS = 20


def blur(x):
    return TF.gaussian_blur(x, kernel_size=51, sigma=20.0)


@torch.no_grad()
def perturbation_curves(net, x, cam, cls, device, seed):
    """Deletion/insertion curves over STEPS equal pixel fractions, plus random-order deletion."""
    n = cam.size
    order = np.argsort(-cam.ravel())
    rand = np.random.default_rng(seed).permutation(n)
    base = blur(x)
    def curve(idx_order, start, end):
        batch = []
        for k in range(STEPS + 1):
            m = torch.zeros(n, device=device)
            m[torch.as_tensor(idx_order[: int(n * k / STEPS)], device=device)] = 1
            m = m.view(1, 1, *cam.shape)
            batch.append(start * (1 - m) + end * m)
        p = F.softmax(net(torch.cat(batch)).float(), -1)[:, cls]
        return float(p.mean())   # area under the curve on a [0, 1] x-axis (equal spacing)
    return {"deletion_auc": curve(order, x, base), "insertion_auc": curve(order, base, x),
            "random_deletion_auc": curve(rand, x, base)}


def main(opts):
    run_folder = UNC / "output" / opts.run
    config = utils.read_config(run_folder)
    net_config = config["Network"]["Basic Setup"]
    assert net_config["network_type"] == "deterministic", "Grad-CAM arm explains the deterministic ResNet50"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_network(config, run_folder)
    load_best(net, run_folder)
    net.to(device).eval()
    target_layer = net.model.layer4[-1]

    preds = pd.read_csv(run_folder / "predictions.csv")
    preds = preds[preds.split.isin(opts.splits)]
    _, test_tf = dataset.medirv2_transforms(net_config["input_size"])
    size = net_config["input_size"]
    out = Path(opts.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    overlay_ids = set(opts.overlay_ids or [])

    rows, cams = [], {}
    cam_engine = GradCAM(model=net, target_layers=[target_layer])
    for n, r in enumerate(preds.itertuples(), 1):
        img = Image.open(Path(opts.dataset_path) / "ISIC_clinical" / f"{r.isic_id}.jpg").convert("RGB")
        x = test_tf(img)[None].to(device)
        cls = int(r.pred)                               # explain the model's own prediction (P1-16)
        cam = cam_engine(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        mask = np.array(Image.open(Path(opts.masks) / f"{r.isic_id}.png").resize((size, size), Image.NEAREST)) > 127
        total = cam.sum()
        row = {"isic_id": r.isic_id, "split": r.split, "pred": cls,
               "cam_in_lesion": float(cam[mask].sum() / total) if total > 0 else np.nan,
               "lesion_area_224": float(mask.mean()),
               "pointing_hit": bool(ndimage.binary_dilation(mask, iterations=5)[np.unravel_index(cam.argmax(), cam.shape)]),
               "cam_entropy": float(-(p := cam.ravel() / max(total, 1e-12))[p > 0].dot(np.log(p[p > 0])))}
        if opts.faithfulness:
            row.update(perturbation_curves(net, x, cam, cls, device, seed=n))
        rows.append(row)
        cams[r.isic_id] = cam.astype(np.float16)
        if r.isic_id in overlay_ids:
            raw = np.asarray(img.resize((size, size))) / 255.0
            Image.fromarray(show_cam_on_image(raw.astype(np.float32), cam, use_rgb=True)).save(
                out / "overlays" / f"{r.isic_id}_gradcam.png")
        if n % 200 == 0:
            print(f"{n}/{len(preds)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gradcam_metrics.csv", index=False)
    np.savez_compressed(out / "cams.npz", isic_id=np.array(list(cams)), cam=np.stack(list(cams.values())))
    print(df.describe().T[["mean", "50%"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--faithfulness", action="store_true")
    ap.add_argument("--overlay-ids", nargs="*")
    main(ap.parse_args())
