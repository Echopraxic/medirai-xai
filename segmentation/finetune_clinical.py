"""
Adapt the ISIC 2018 ResNet18-UNet to clinical close-ups with UNet-SAM agreement pseudo-labels (plan T1.2).

Pseudo-labels come from split-v2 train images only, where the UNet and zero-shot SAM masks agree
(Dice >= --min-agreement). Pixels where the two disagree are ignored in the loss, so the model learns the
clinical domain from the confident part of each image without being told which method was right. Each batch
mixes clinical pseudo-labelled images with ISIC 2018 dermoscopy (true masks) so the model does not forget.

Known bias: agreement selects the easier lesions; the gold set (segmentation/gold, hand-annotated) is the
only fair test, and segmentation/gold_eval.py decides between this model and the original.

    python segmentation/finetune_clinical.py --init segmentation/output/unet_r18/best.pt \
        --isic2018 uncertaintyNet-main/datasets_768/ISIC2018 --out segmentation/output/unet_r18_clinical
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from train_unet import ISIC2018Seg
from unet_resnet import IMAGENET_MEAN, IMAGENET_STD, ResNetUNet, dice_coefficient

ROOT = Path(__file__).resolve().parents[1]
CLIN = ROOT / "uncertaintyNet-main/datasets_768"


class PseudoLabelled(Dataset):
    """Clinical image + 3-level target: 1 = both masks lesion, 0 = both background, 2 = disagree (ignored)."""

    def __init__(self, ids, size, train):
        self.ids = ids
        norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        aug = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
               v2.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
               v2.Resize((size, size), antialias=True),
               v2.RandomApply([v2.ColorJitter(0.25, 0.25, 0.25, 0.04)], p=0.8)] if train else \
              [v2.Resize((size, size), antialias=True)]
        self.tf = v2.Compose([v2.ToImage(), *aug, *norm])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        isic_id = self.ids[i]
        img = Image.open(CLIN / "ISIC_clinical" / f"{isic_id}.jpg").convert("RGB")
        a = np.array(Image.open(CLIN / "ISIC_clinical_masks/unet" / f"{isic_id}.png")) > 127
        b = np.array(Image.open(CLIN / "ISIC_clinical_masks/sam" / f"{isic_id}.png")) > 127
        target = np.where(a & b, 1, np.where(~a & ~b, 0, 2)).astype(np.uint8)
        img, target = self.tf(img, tv_tensors.Mask(torch.from_numpy(target)[None]))
        return img, target.float()


def masked_loss(logits, target):
    """BCE + soft Dice over pixels that are not 'ignore' (2). ISIC 2018 targets have no 2s."""
    valid = (target != 2).float()
    t = (target == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    soft_dice = dice_coefficient(p, t * valid).mean()
    return bce + (1 - soft_dice)


@torch.no_grad()
def val_dice(net, loader, device):
    net.eval()
    out = []
    for img, target in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = (torch.sigmoid(net(img.to(device)).float()) > 0.5).float()
        valid = (target != 2).float().to(device)
        out.append(dice_coefficient(pred * valid, (target == 1).float().to(device) * valid).cpu())
    return float(torch.cat(out).mean())


def main(opts):
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    qa = pd.read_csv(ROOT / "segmentation/output/qa/mask_quality.csv")[["isic_id", "unet_sam_dice"]]
    v2_split = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")[["isic_id", "split"]]
    qa = qa.merge(v2_split, on="isic_id")
    gold = set(pd.read_csv(ROOT / "segmentation/gold/gold_set.csv").isic_id)
    agree = qa[(qa.unet_sam_dice >= opts.min_agreement) & ~qa.isic_id.isin(gold)]
    tr_ids, va_ids = agree[agree.split == "train"].isic_id.tolist(), agree[agree.split == "val"].isic_id.tolist()
    print(f"pseudo-labelled: train {len(tr_ids)}, val {len(va_ids)} (agreement >= {opts.min_agreement})")

    isic = ISIC2018Seg(Path(opts.isic2018), "train", opts.size, True)
    train_ds = ConcatDataset([PseudoLabelled(tr_ids, opts.size, True), isic])
    g = torch.Generator().manual_seed(opts.seed)
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers,
                          drop_last=True, generator=g, persistent_workers=opts.workers > 0)
    clin_val = DataLoader(PseudoLabelled(va_ids, opts.size, False), batch_size=16, num_workers=opts.workers)
    derm_val = DataLoader(ISIC2018Seg(Path(opts.isic2018), "val", opts.size, False), batch_size=16,
                          num_workers=opts.workers)

    ckpt = torch.load(opts.init, map_location=device)
    net = ResNetUNet(pretrained=False).to(device)
    net.load_state_dict(ckpt["state_dict"])
    base = {"clinical_pseudo_val_dice": val_dice(net, clin_val, device), "isic2018_val_dice": val_dice(net, derm_val, device)}
    print("before", json.dumps(base), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=opts.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opts.lr, total_steps=opts.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, log = -1.0, []
    for epoch in range(opts.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for img, target in train_dl:
            img, target = img.to(device), target.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = net(img)
            loss = masked_loss(logits.float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        row = {"epoch": epoch, "train_loss": total / len(train_dl),
               "clinical_pseudo_val_dice": val_dice(net, clin_val, device),
               "isic2018_val_dice": val_dice(net, derm_val, device), "seconds": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        score = row["clinical_pseudo_val_dice"] + row["isic2018_val_dice"]   # do not trade one domain for the other
        if score > best:
            best = score
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "size": opts.size, "val": row,
                        "init": str(opts.init)}, out / "best.pt")
    (out / "summary.json").write_text(json.dumps({"before": base, "log": log, "args": vars(opts),
                                                  "n_pseudo_train": len(tr_ids), "n_pseudo_val": len(va_ids)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--isic2018", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-agreement", type=float, default=0.85)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
