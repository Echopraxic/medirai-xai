"""
Classifier sprint stage B batch 2 trainer (prereg 2026-10-21; decisions 2026-10-21: ResNet50 is the product model,
DINOv2 a comparison). One binary benign/malignant model per run, same split files and image cache as train.py.

Options (each batch-2 arm changes one thing against the hygiene recipe):
  --backbone resnet50   uncertaintyNet's ImageNet ResNet50 (the product model)       | dinov2: facebook/dinov2-large
                        fine-tuned with a linear head on CLS (comparison baseline only)
  hygiene               --optimizer adamw --label_smoothing 0.1 --ema 0.999 (EMA weights are evaluated and saved)
  --input_size 384      higher resolution (B1)
  --distill W           B4: frozen DINOv2-L teacher on the same augmented batch; loss += W * (1 - cos(proj(student
                        penultimate), teacher CLS)). The projection is trained and discarded; the product is a ResNet50.
  --teacher_runs A,B,.. stage C: distil an ensemble of trained ResNet50 runs into this one (Hinton KD; --kd_alpha,
                        --kd_T); the teacher sees the same augmented batch, including unlabelled-in-practice MIDAS photos
  --mel_head W          melanoma head (decision 2026-10-22): a second output p(melanoma) on the penultimate features,
                        loss += W * BCE (pos_weight = sqrt(neg/pos) on train); enables the melanoma-sensitive mode
  --photo_aug           B2: phone degradations (JPEG 30-90, blur, exposure) + melanin-like darkening (blue reduced
                        more than red; heuristic, applied to the whole image)
  --pairs_csv F --consistency W
                        arm S4 (prereg 2026-10-23): light/dark image pairs (columns orig_id, synth_id; e.g. the Mendeley
                        I2SB pairs) seen with the same augmentation; loss += W * symmetric KL between the two softmax
                        outputs. Pairs carry no label or KD loss; they share the main batch's forward pass (BatchNorm)
Checkpoint = best val AUROC (val = the split's val rows: split v2 val, or v2 val + MIDAS dev_val for split v4).
Writes the same files as uncertaintyNet runs (checkpoint_best_val.pth.tar, config.json, log.csv, predictions.csv,
embeddings.npz), so eval/selection_score.py, trees/, eval/gradcam_arm.py and interpret/ read it unchanged.

    python classifier/train_sprint.py --dataset-path <folder with ISIC_clinical/> --out <run dir> \
        --optimizer adamw --label_smoothing 0.1 --ema 0.999 [--input_size 384] [--distill 1.0] [--photo_aug]
"""
import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uncertaintyNet-main"))
import dataset  # noqa: E402

MEDIR_STATS = ([0.6689, 0.5090, 0.4417], [0.1336, 0.1352, 0.1486])   # uncertaintyNet MEDIRV2 normalization
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])       # DINOv2
DINOV2 = "facebook/dinov2-large"


# ------------------------------------------------------------------------------------------------ augmentation
class PhoneDegrade:
    """JPEG compression, defocus blur and exposure changes typical of smartphone photos."""
    def __call__(self, im):
        if random.random() < 0.5:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=random.randint(30, 90))
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return im


class MelaninDarken:
    """Heuristic darker-skin simulation on the tensor in [0, 1]: scale RGB with blue reduced most (melanin absorbs
    short wavelengths more). Whole image, so the lesion darkens too; a coarse proxy, tested as its own arm."""
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        a = random.uniform(0.45, 0.85)
        return (x * torch.tensor([a, a * 0.9, a * 0.8]).view(3, 1, 1)).clamp(0, 1)


class PairDataset(torch.utils.data.Dataset):
    """Tone pairs for the consistency term: both images get the same random augmentation (python and torch RNG seeded
    identically inside a forked RNG state, so the worker's own streams are left untouched)."""
    def __init__(self, pairs, data_dir, tf):
        self.pairs, self.data_dir, self.tf = list(zip(pairs.orig_id, pairs.synth_id)), Path(data_dir), tf

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)))
        out = []
        for i in self.pairs[k]:
            im = Image.open(self.data_dir / f"{i}.jpg").convert("RGB")
            state = random.getstate()
            with torch.random.fork_rng(devices=[]):
                random.seed(seed)
                torch.manual_seed(seed)
                out.append(self.tf(im))
            random.setstate(state)
        return out[0], out[1]


def forever(loader):
    while True:
        yield from loader


def make_transforms(input_size, stats, photo_aug=False):
    norm = transforms.Normalize(*stats)
    train = [transforms.Resize((input_size, input_size))]
    if photo_aug:
        train.insert(0, PhoneDegrade())
    train += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(60),
              transforms.ColorJitter(brightness=0.3 if photo_aug else 0.2, contrast=0.3 if photo_aug else 0.2,
                                     saturation=0.2, hue=0.1),
              transforms.ToTensor()]
    if photo_aug:
        train.append(MelaninDarken())
    train.append(norm)
    test = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), norm])
    return transforms.Compose(train), test


# ------------------------------------------------------------------------------------------------ models
class ResNetBinary(nn.Module):
    """uncertaintyNet's ResNet50 (same weights init and layout as train.py's det runs); exposes the penultimate.
    mel_head: an extra Linear(2048 -> 1) melanoma logit on the penultimate features (melanoma-sensitive mode)."""
    def __init__(self, mel_head=False):
        super().__init__()
        from models.model import get_model
        cfg = json.loads((ROOT / "uncertaintyNet-main/configs/medir_v2.json").read_text())["Network"]["Basic Setup"]
        self.net = get_model(dict(cfg, output_size=2, network_type="deterministic"))
        self.pool = next(m for n, m in self.net.named_modules() if n.endswith("avgpool"))
        self.feat_dim = 2048
        self.mel_head = nn.Linear(self.feat_dim, 1) if mel_head else None

    def forward(self, x):
        feats = []  # hook registered per call, so deep copies (EMA) never write into another module
        handle = self.pool.register_forward_hook(lambda m, i, o: feats.append(o.flatten(1)))
        try:
            logits = self.net(x)
        finally:
            handle.remove()
        return logits, feats[0]


class DinoBinary(nn.Module):
    def __init__(self, local_files_only=False):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(DINOV2, local_files_only=local_files_only)
        self.feat_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(self.feat_dim, 2)

    def forward(self, x):
        f = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head(f), f


def build_model(backbone, local_files_only=False, mel_head=False):
    if backbone != "resnet50" and mel_head:
        raise ValueError("the melanoma head is implemented for the ResNet50 product model only")
    return ResNetBinary(mel_head) if backbone == "resnet50" else DinoBinary(local_files_only)


def melanoma_prob(model, feats):
    return torch.sigmoid(model.mel_head(feats.float()).squeeze(1)) if getattr(model, "mel_head", None) is not None else None


class Teacher(nn.Module):
    """Frozen DINOv2-L CLS embedding of the student's (augmented) batch, re-normalized to ImageNet stats at 224 px."""
    def __init__(self, student_stats, local_files_only):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(DINOV2, local_files_only=local_files_only).eval().requires_grad_(False)
        s_mean, s_std = (torch.tensor(v).view(1, 3, 1, 1) for v in student_stats)
        t_mean, t_std = (torch.tensor(v).view(1, 3, 1, 1) for v in IMAGENET_STATS)
        for k, v in dict(s_mean=s_mean, s_std=s_std, t_mean=t_mean, t_std=t_std).items():
            self.register_buffer(k, v)

    @torch.no_grad()
    def forward(self, x):
        x = ((x * self.s_std + self.s_mean) - self.t_mean) / self.t_std
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return self.m(pixel_values=x).last_hidden_state[:, 0].float()


class EnsembleTeacher(nn.Module):
    """Frozen ensemble of trained ResNet50 runs (stage C, decision 2026-10-22: distil the ensemble into one ResNet50).
    Returns the mean of the members' temperature-softened probabilities on the student's own (augmented) batch.
    Members must share the student's normalization and input size (checked)."""
    def __init__(self, runs, input_size):
        super().__init__()
        sys.path.insert(0, str(ROOT / "eval"))
        from selection_score import load_net
        self.members, self.kinds = nn.ModuleList(), []
        for run in runs:
            run = Path(run) if Path(run).is_absolute() else ROOT / "uncertaintyNet-main/output" / run
            cfg = json.loads((run / "config.json").read_text())
            size = cfg.get("input_size") or cfg["Network"]["Basic Setup"]["input_size"]
            if size != input_size or cfg.get("normalization", "medir") != "medir":
                raise ValueError(f"teacher {run} does not match the student's input ({size} px, {cfg.get('normalization')})")
            net, _, kind = load_net(run)
            self.members.append(net.eval().requires_grad_(False))
            self.kinds.append(kind)

    @torch.no_grad()
    def forward(self, x, T):
        probs = []
        for net, kind in zip(self.members, self.kinds):
            logits = net(x)[0] if kind == "sprint" else net(x)
            probs.append(F.softmax(logits.float() / T, dim=1))
        return torch.stack(probs).mean(0)


def kd_loss(student_logits, teacher_probs, T):
    """Hinton et al. 2015: T^2 * KL(teacher || student) on temperature-softened distributions."""
    log_s = F.log_softmax(student_logits.float() / T, dim=1)
    return F.kl_div(log_s, teacher_probs, reduction="batchmean") * T * T


def symmetric_kl(a, b):
    """0.5 * (KL(p_a || p_b) + KL(p_b || p_a)) on softmax outputs (T = 1), batch mean."""
    la, lb = F.log_softmax(a.float(), dim=1), F.log_softmax(b.float(), dim=1)
    return 0.5 * (F.kl_div(lb, la, log_target=True, reduction="batchmean")
                  + F.kl_div(la, lb, log_target=True, reduction="batchmean"))


# ------------------------------------------------------------------------------------------------ train / eval
@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    probs, feats, ids, labels, mels = [], [], [], [], []
    for x, y, i in loader:
        with torch.autocast(device.type, enabled=amp):
            logits, f = model(x.to(device, non_blocking=True))
        pm = melanoma_prob(model, f)
        if pm is not None:
            mels.append(pm.float().cpu().numpy())
        probs.append(F.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        feats.append(f.float().cpu().numpy().astype(np.float16))
        ids += list(i)
        labels.append(y.numpy())
    predict.last_mel = np.concatenate(mels) if mels else None   # p(melanoma) of the last call, if the model has the head
    return np.concatenate(probs), np.concatenate(feats), ids, np.concatenate(labels)


def main(opts):
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data_dir = str(Path(opts.dataset_path) / "ISIC_clinical")
    df = dataset.load_split(str(ROOT / "splits" / opts.split_file), data_dir)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if opts.limit:
        parts = {s: d.groupby("label", group_keys=False).head(opts.limit // 2).reset_index(drop=True)
                 for s, d in parts.items()}
    stats = MEDIR_STATS if opts.backbone == "resnet50" else IMAGENET_STATS
    train_tf, test_tf = make_transforms(opts.input_size, stats, opts.photo_aug)
    g = torch.Generator().manual_seed(opts.seed)
    kw = dict(num_workers=opts.workers, pin_memory=amp, persistent_workers=opts.workers > 0)
    train_dl = DataLoader(dataset.ISICDatasetV2(parts["train"], data_dir, train_tf), batch_size=opts.batch_size,
                          shuffle=True, drop_last=True, generator=g, **kw)
    eval_dl = {s: DataLoader(dataset.ISICDatasetV2(parts[s], data_dir, test_tf), batch_size=opts.batch_size, **kw)
               for s in parts}
    pair_iter = None
    if opts.consistency > 0:
        pairs = pd.read_csv(ROOT / "splits" / opts.pairs_csv)
        if opts.limit:
            pairs = pairs.head(opts.limit)
        pair_iter = forever(DataLoader(PairDataset(pairs, data_dir, train_tf), batch_size=opts.pair_batch, shuffle=True,
                                       drop_last=True, generator=torch.Generator().manual_seed(opts.seed + 1), **kw))

    model = build_model(opts.backbone, opts.offline, opts.mel_head > 0).to(device)
    is_mel = {i: "Melanoma" in str(d) for i, d in zip(parts["train"].isic_id, parts["train"].diagnosis_3)}
    n_mel = max(sum(is_mel.values()), 1)
    mel_pos_weight = torch.tensor(((len(is_mel) - n_mel) / n_mel) ** 0.5, device=device)
    teacher = proj = ens_teacher = None
    params = [{"params": model.parameters(), "lr": opts.lr}]
    if opts.backbone == "dinov2":  # small LR for the pretrained ViT, larger for the new head
        params = [{"params": model.backbone.parameters(), "lr": opts.lr},
                  {"params": model.head.parameters(), "lr": opts.lr * 50}]
    if opts.distill > 0:
        teacher = Teacher(stats, opts.offline).to(device)
        proj = nn.Linear(model.feat_dim, teacher.m.config.hidden_size).to(device)
        params.append({"params": proj.parameters(), "lr": opts.lr})
    if opts.teacher_runs:
        ens_teacher = EnsembleTeacher(opts.teacher_runs.split(","), opts.input_size).to(device)
    opt = (torch.optim.AdamW(params, weight_decay=opts.weight_decay) if opts.optimizer == "adamw"
           else torch.optim.Adam(params, weight_decay=opts.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # use_buffers: BatchNorm running stats are averaged too (otherwise the EMA model keeps its initial BN stats)
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(opts.ema),
                                               use_buffers=True) if opts.ema > 0 else None)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], -1.0
    for epoch in range(1, opts.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, ids in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                if pair_iter is not None:  # one forward pass: labelled batch, then the light and the dark half of the pairs
                    xo, xs = (t.to(device, non_blocking=True) for t in next(pair_iter))
                    logits_all, f_all = model(torch.cat([x, xo, xs]))
                    n0, n1 = len(x), len(x) + len(xo)
                    logits, f, pair_o, pair_s = logits_all[:n0], f_all[:n0], logits_all[n0:n1], logits_all[n1:]
                else:
                    logits, f = model(x)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=opts.label_smoothing)
                if opts.mel_head > 0:
                    target = torch.tensor([float(is_mel[i]) for i in ids], device=device)
                    loss = loss + opts.mel_head * F.binary_cross_entropy_with_logits(
                        model.mel_head(f.float()).squeeze(1), target, pos_weight=mel_pos_weight)
                if ens_teacher is not None:
                    loss = (1 - opts.kd_alpha) * loss + opts.kd_alpha * kd_loss(logits, ens_teacher(x, opts.kd_T), opts.kd_T)
                if teacher is not None:
                    loss = loss + opts.distill * (1 - F.cosine_similarity(proj(f.float()), teacher(x), dim=1)).mean()
                if pair_iter is not None:
                    loss = loss + opts.consistency * symmetric_kl(pair_o, pair_s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            tot, n = tot + loss.item() * len(y), n + len(y)
        sched.step()
        evaluated = ema.module if ema is not None else model
        p, _, _, yv = predict(evaluated, eval_dl["val"], device, amp)
        auc = roc_auc_score(yv, p)
        log.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_auroc": auc, "seconds": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if auc > best:
            best = auc
            torch.save({"epoch": epoch, "network_state_dict": evaluated.state_dict()}, out / "checkpoint_best_val.pth.tar")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

    final = build_model(opts.backbone, opts.offline, opts.mel_head > 0).to(device)
    state = torch.load(out / "checkpoint_best_val.pth.tar", map_location=device, weights_only=False)
    final.load_state_dict(state["network_state_dict"])
    frames, embs, emb_ids = [], [], []
    for s in ["train", "val", "test"]:
        p, f, ids, y = predict(final, eval_dl[s], device, amp)
        p_mel = predict.last_mel
        pc = p.clip(1e-7, 1 - 1e-7)
        ent = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc))
        frames.append(pd.DataFrame({"isic_id": ids, "label": y.astype(int), "pred": (p >= 0.5).astype(int),
                                    "p_malignant": p, "logit_0": np.log(1 - pc), "logit_1": np.log(pc),
                                    "max_softmax": np.maximum(p, 1 - p), "entropy_of_expected": ent,
                                    "expected_entropy": ent, "mutual_information": 0.0, "mc_std_p_malignant": 0.0,
                                    "split": s, "diagnosis_3": parts[s].diagnosis_3.to_numpy(),
                                    **({"p_melanoma": p_mel} if p_mel is not None else {})}))
        embs.append(f)
        emb_ids += ids
    pd.concat(frames).to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(out / "embeddings.npz", isic_id=np.array(emb_ids), emb=np.concatenate(embs))
    cfg = {"trainer": "train_sprint", "backbone": opts.backbone, "input_size": opts.input_size, "mel_head": opts.mel_head > 0,
           "normalization": "medir" if opts.backbone == "resnet50" else "imagenet", "malignant_index": 1,
           "best_epoch": state["epoch"], "best_val_auroc": best, **vars(opts)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "predictions_meta.json").write_text(json.dumps({"run": str(out), "best_epoch": state["epoch"],
                                                           "network_type": f"sprint:{opts.backbone}", "seed": opts.seed}))
    print(f"best epoch {state['epoch']} val AUROC {best:.4f} -> {out}")


def load_for_inference(run, device="cpu"):
    """(model returning (logits, features), test transform) for a train_sprint run folder."""
    cfg = json.loads((Path(run) / "config.json").read_text())
    model = build_model(cfg["backbone"], local_files_only=True, mel_head=cfg.get("mel_head", False))
    state = torch.load(Path(run) / "checkpoint_best_val.pth.tar", map_location=device, weights_only=False)
    model.load_state_dict(state["network_state_dict"])
    stats = MEDIR_STATS if cfg["normalization"] == "medir" else IMAGENET_STATS
    return model.eval(), make_transforms(cfg["input_size"], stats)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_file", default="isic_clinical_v2.csv")
    ap.add_argument("--backbone", choices=["resnet50", "dinov2"], default="resnet50")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off)")
    ap.add_argument("--distill", type=float, default=0.0, help="weight of the DINOv2 feature-distillation loss")
    ap.add_argument("--teacher_runs", default="", help="ensemble distillation: comma-separated run folders (ResNet50)")
    ap.add_argument("--kd_alpha", type=float, default=0.7, help="weight of the ensemble KD loss vs the label loss")
    ap.add_argument("--kd_T", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--photo_aug", action="store_true")
    ap.add_argument("--mel_head", type=float, default=0.0, help="weight of the melanoma-head BCE loss (0 = no head)")
    ap.add_argument("--pairs_csv", default="mendeley_pairs_v1.csv", help="tone pairs under splits/ (orig_id, synth_id)")
    ap.add_argument("--consistency", type=float, default=0.0, help="weight of the tone-consistency loss (0 = off)")
    ap.add_argument("--pair_batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true", help="load DINOv2 from the HF cache only (cluster nodes)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many images per split")
    main(ap.parse_args())
