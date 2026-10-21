"""
ResNet50 with a diagnosis-class head instead of benign/malignant (classifier improvement, 2026-10-15).

The binary model sees melanoma only as part of "malignant" (BCC is 4x more common), and its melanoma recall at the
default threshold is ~0.5. This trains the same backbone, preprocessing and split (split v2, uncertaintyNet's
ResNet50Hidden + medirv2_transforms; --split_file for split v4) on 7 classes, with class-weighted cross-entropy (inverse sqrt frequency):

    BCC, SCC, Melanoma, Other-malignant | Nevus, SK, Other-benign

The one-vs-all "Other" bucket is split by label so p(malignant) is exactly the summed malignant-class probability.
The checkpoint is selected on val AUROC of p(malignant). predictions.csv has the same columns as
eval/export_predictions.py (binary entropy of p(malignant) as entropy_of_expected), plus prob_<class> and pred_class,
so eval/baseline_report.py, eval/calibrate_ensemble.py and trees/train_trees.py take it unchanged.

    python classifier/train_multiclass.py --dataset-path uncertaintyNet-main/datasets_768 --out classifier/output/mc_run \
        [--batch_size 32 --grad_accum 4 --amp]       # laptop settings; cluster: defaults (batch 128)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uncertaintyNet-main"))
sys.path.insert(0, str(ROOT / "features"))
import dataset  # noqa: E402
from models.model import get_model  # noqa: E402
from build_table import OVA_MAP  # noqa: E402

CLASSES = ["BCC", "SCC", "Melanoma", "Other-malignant", "Nevus", "SK", "Other-benign"]
MALIGNANT = [0, 1, 2, 3]


def class_of(diagnosis_3, label):
    c = OVA_MAP.get(diagnosis_3, "Other")
    return f"Other-{'malignant' if label == 1 else 'benign'}" if c == "Other" else c


def class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / (w * counts).sum() * counts.sum(), dtype=torch.float32)


def malignant_prob(logits):
    return F.softmax(logits.float(), dim=1)[:, MALIGNANT].sum(1)


class DiagnosisDataset(dataset.ISICDatasetV2):
    def __init__(self, df, root_dir, transform):
        super().__init__(df, root_dir, transform)
        self.labels = torch.tensor([CLASSES.index(class_of(d, int(l))) for d, l in zip(df.diagnosis_3, df.label)])


@torch.no_grad()
def predict(net, loader, device, amp):
    net.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast("cuda", enabled=amp):
            out.append(net(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def main(opts):
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((ROOT / "uncertaintyNet-main/configs/medir_v2.json").read_text())
    data_dir = str(Path(opts.dataset_path) / "ISIC_clinical")
    df = dataset.load_split(str(ROOT / "splits" / opts.split_file), data_dir)
    train_tf, test_tf = dataset.medirv2_transforms(224)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if opts.limit:
        parts = {s: d.groupby("label", group_keys=False).head(opts.limit // 2) for s, d in parts.items()}
    train_ds = DiagnosisDataset(parts["train"], data_dir, train_tf)
    kw = dict(num_workers=opts.workers, pin_memory=device.type == "cuda", persistent_workers=opts.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True, drop_last=True, **kw)
    eval_dl = {s: DataLoader(DiagnosisDataset(parts[s], data_dir, test_tf), batch_size=opts.batch_size, **kw)
               for s in ["val", "test"]}

    net_cfg = dict(cfg["Network"]["Basic Setup"], output_size=len(CLASSES), network_type="deterministic")
    net = get_model(net_cfg).to(device)
    weights = class_weights(train_ds.labels.numpy()).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opts.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=opts.amp)
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best = [], -1.0
    y_val = parts["val"].label.values
    for epoch in range(1, opts.epochs + 1):
        net.train()
        t0, total, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(train_dl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=opts.amp):
                loss = F.cross_entropy(net(x).float(), y, weight=weights) / opts.grad_accum
            scaler.scale(loss).backward()
            if step % opts.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total, n = total + loss.item() * opts.grad_accum * len(y), n + len(y)
        sched.step()
        logits = predict(net, eval_dl["val"], device, opts.amp)
        auc = roc_auc_score(y_val, malignant_prob(logits).numpy())
        cls_acc = (logits.argmax(1) == eval_dl["val"].dataset.labels).float().mean().item()
        log.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_auroc_malignant": auc,
                    "val_class_accuracy": cls_acc, "seconds": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if auc > best:
            best = auc
            torch.save({"epoch": epoch, "network_state_dict": net.state_dict(), "classes": CLASSES},
                       out / "checkpoint_best_val.pth.tar")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

    state = torch.load(out / "checkpoint_best_val.pth.tar", map_location=device, weights_only=False)
    net.load_state_dict(state["network_state_dict"])
    rows = []
    for s in ["train", "val", "test"]:
        dl = DataLoader(DiagnosisDataset(parts[s], data_dir, test_tf), batch_size=opts.batch_size, **kw)
        logits = predict(net, dl, device, opts.amp)
        prob = F.softmax(logits, dim=1).numpy()
        p = prob[:, MALIGNANT].sum(1).clip(1e-7, 1 - 1e-7)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        d = parts[s]
        frame = pd.DataFrame({"isic_id": d.isic_id, "label": d.label.astype(int), "pred": (p >= 0.5).astype(int),
                              "p_malignant": p, "logit_0": np.log(1 - p), "logit_1": np.log(p),
                              "max_softmax": np.maximum(p, 1 - p), "entropy_of_expected": ent, "expected_entropy": ent,
                              "mutual_information": 0.0, "mc_std_p_malignant": 0.0, "split": s,
                              "diagnosis_3": d.diagnosis_3, "pred_class": np.array(CLASSES)[prob.argmax(1)]})
        for i, c in enumerate(CLASSES):
            frame[f"prob_{c}"] = prob[:, i]
        rows.append(frame)
    pd.concat(rows).to_csv(out / "predictions.csv", index=False)
    meta = {"run": str(out), "best_epoch": state["epoch"], "network_type": "deterministic_multiclass",
            "mc_samples": 1, "seed": opts.seed, "dataset_path": str(opts.dataset_path), "n": int(sum(map(len, rows)))}
    (out / "predictions_meta.json").write_text(json.dumps(meta, indent=1))
    (out / "config.json").write_text(json.dumps({"classes": CLASSES, "class_weights": weights.tolist(),
                                                 "best_epoch": state["epoch"], "best_val_auroc": best, **vars(opts)},
                                                indent=1))
    print(f"best epoch {state['epoch']} val AUROC {best:.4f}; wrote {out / 'predictions.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True, help="folder that contains ISIC_clinical/<isic_id>.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: this many images per split")
    ap.add_argument("--split_file", default="isic_clinical_v2.csv", help="file under splits/ (stage B: v4)")
    main(ap.parse_args())
