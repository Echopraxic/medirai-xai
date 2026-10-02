"""
Train a ResNet18-UNet lesion segmenter on ISIC 2018 Task 1 (dermoscopic) for use on ISIC clinical close-ups.

The target domain differs (clinical photos: lesion smaller in frame, no dermoscope vignette, different
lighting), so augmentation leans on zoom-out, color/gamma/blur jitter. Model selection uses the ISIC 2018
validation set; the 1,000-image test set is scored once at the end (Dice, Jaccard, thresholded Jaccard).

    python segmentation/train_unet.py --data uncertaintyNet-main/datasets_768/ISIC2018 --out segmentation/output/unet_r18
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from unet_resnet import IMAGENET_MEAN, IMAGENET_STD, ResNetUNet, dice_coefficient, jaccard


class ISIC2018Seg(Dataset):
    def __init__(self, root: Path, part: str, size: int, train: bool):
        self.imgs = sorted((root / part / "img").glob("*.jpg"))
        self.segs = [root / part / "seg" / p.name.replace(".jpg", "_segmentation.png") for p in self.imgs]
        assert self.imgs and all(s.exists() for s in self.segs), f"missing masks under {root / part}"
        norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if train:
            self.tf = v2.Compose([
                v2.ToImage(),
                v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                # zoom out (lesion smaller in frame, as in clinical close-ups) as well as in
                v2.RandomAffine(degrees=180, translate=(0.15, 0.15), scale=(0.45, 1.25), shear=8),
                v2.Resize((size, size), antialias=True),
                v2.RandomApply([v2.ColorJitter(0.35, 0.35, 0.35, 0.08)], p=0.9),
                v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
                v2.RandomGrayscale(p=0.05),
                *norm,
            ])
        else:
            self.tf = v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True), *norm])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = tv_tensors.Mask(torch.from_numpy((np.array(Image.open(self.segs[i])) > 127).astype(np.uint8))[None])
        img, mask = self.tf(img, mask)
        return img, mask.float()


def loss_fn(logits, target):
    prob = torch.sigmoid(logits)
    soft_dice = dice_coefficient(prob, target).mean()
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - soft_dice)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    dices, jacs = [], []
    for img, mask in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = net(img.to(device))
        pred = (torch.sigmoid(logits.float()) > 0.5).float()
        mask = mask.to(device)
        dices.append(dice_coefficient(pred, mask).cpu())
        jacs.append(jaccard(pred, mask).cpu())
    d, j = torch.cat(dices), torch.cat(jacs)
    # ISIC 2018 challenge metric: per-image Jaccard, set to 0 when below 0.65
    return {"dice": float(d.mean()), "jaccard": float(j.mean()),
            "thresholded_jaccard": float(torch.where(j < 0.65, torch.zeros_like(j), j).mean())}


def main(opts):
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(opts.data)
    g = torch.Generator().manual_seed(opts.seed)
    train_dl = DataLoader(ISIC2018Seg(root, "train", opts.size, True), batch_size=opts.batch_size, shuffle=True,
                          num_workers=opts.workers, drop_last=True, generator=g, persistent_workers=opts.workers > 0)
    val_dl = DataLoader(ISIC2018Seg(root, "val", opts.size, False), batch_size=16, num_workers=opts.workers)
    test_dl = DataLoader(ISIC2018Seg(root, "test", opts.size, False), batch_size=16, num_workers=opts.workers)

    net = ResNetUNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=opts.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opts.lr, total_steps=opts.epochs * len(train_dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, log = -1.0, []
    for epoch in range(opts.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = net(img)
            loss = loss_fn(logits.float(), mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
        val = evaluate(net, val_dl, device)
        row = {"epoch": epoch, "train_loss": total / len(train_dl), **{f"val_{k}": v for k, v in val.items()},
               "seconds": time.time() - t0}
        log.append(row)
        print(json.dumps(row), flush=True)
        if val["dice"] > best:
            best = val["dice"]
            torch.save({"state_dict": net.state_dict(), "epoch": epoch, "size": opts.size, "val": val}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    test = evaluate(net, test_dl, device)
    summary = {"best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": test, "args": vars(opts)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print("TEST", json.dumps(test))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
