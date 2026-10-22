"""
Pre-registered selection score for one run (classifier sprint, prereg 2026-10-21): reads only the selection data,
split v2 val (internal) and MIDAS dev_val phone photos. Run at the end of every stage-B job.

  S      = mean(internal val AUROC, MIDAS dev_val AUROC), with 8-view TTA (adopted in stage A1); S without TTA too
  at the internal-val threshold t giving malignant sensitivity >= 0.90 (TTA): internal val sensitivity, specificity
  and melanoma recall (the prereg's no-worse-than-0.02 checks), plus MIDAS dev_val sensitivity and specificity

Binary uncertaintyNet runs (config.json + checkpoint_best_val.pth.tar), 7-class runs from
classifier/train_multiclass.py (p(malignant) = summed malignant-class probability) and classifier/train_sprint.py
runs (ResNet50 or DINOv2) are accepted.

    python eval/selection_score.py --run uncertaintyNet-main/output/<tag>/resnet50_det/run \
        --isic_dir uncertaintyNet-main/datasets_768/ISIC_clinical --midas_dir data/midas/files_768
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))

import dataset  # noqa: E402
import utils  # noqa: E402
from preprocess_ablation import DIHEDRAL  # noqa: E402

MALIGNANT_CLASSES = 4  # train_multiclass.CLASSES: the first four are malignant


def load_net(run):
    """(net, test transform, kind): kind 'multiclass' (7-class head), 'sprint' (classifier/train_sprint.py: returns
    (logits, features), may use ImageNet normalization) or 'binary' (uncertaintyNet run)."""
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    if cfg.get("trainer") == "train_sprint":
        sys.path.insert(0, str(ROOT / "classifier"))
        from train_sprint import load_for_inference
        net, tf = load_for_inference(run)
        return net, tf, "sprint"
    state = torch.load(run / "checkpoint_best_val.pth.tar", map_location="cpu", weights_only=False)
    if "classes" in cfg:  # 7-class head
        from models.model import get_model
        base = json.loads((UNC / "configs/medir_v2.json").read_text())["Network"]["Basic Setup"]
        net = get_model(dict(base, output_size=len(cfg["classes"]), network_type="deterministic"))
        net.load_state_dict(state["network_state_dict"])
        return net, dataset.medirv2_transforms(base["input_size"])[1], "multiclass"
    from export_predictions import build_network
    config = utils.read_config(run)
    net = build_network(config, run)
    net.load_state_dict(state["network_state_dict"], strict=True)
    return net, dataset.medirv2_transforms(config["Network"]["Basic Setup"]["input_size"])[1], "binary"


@torch.no_grad()
def p_malignant(net, kind, tf, paths, device, tta, batch_size=32):
    out, mel = [], []
    logits_of = (lambda x: net(x)[0]) if kind == "sprint" else net
    has_mel = kind == "sprint" and getattr(net, "mel_head", None) is not None
    for i in range(0, len(paths), batch_size):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        views = DIHEDRAL if tta else DIHEDRAL[:1]
        probs = torch.stack([F.softmax(logits_of(v(x)).float(), -1) for v in views]).mean(0)
        out.append((probs[:, :MALIGNANT_CLASSES].sum(1) if kind == "multiclass" else probs[:, 1]).cpu().numpy())
        if has_mel:
            mel.append(torch.stack([torch.sigmoid(net.mel_head(net(v(x))[1].float()).squeeze(1)) for v in views]).mean(0).cpu().numpy())
    p_malignant.last_mel = np.concatenate(mel) if mel else None
    return np.concatenate(out)


def melanoma_mode(y, p, pm, mel, t, target=0.65):
    """Melanoma-sensitive mode: malignant if p >= t OR p(melanoma) >= tm, with tm the highest-specificity threshold
    reaching melanoma recall >= target on internal val (chosen there, applied unchanged elsewhere)."""
    best = None
    for tm in np.quantile(pm, np.linspace(0.3, 0.995, 140)):
        pred = (p >= t) | (pm >= tm)
        if pred[mel].mean() >= target and (best is None or (~pred[y == 0]).mean() > best[1]):
            best = (float(tm), float((~pred[y == 0]).mean()))
    return best[0] if best else None


def threshold_for_sensitivity(y, p, target=0.90):
    """Largest threshold whose sensitivity on (y, p) is >= target."""
    pos = np.sort(p[y == 1])
    return float(pos[int(np.floor((1 - target) * len(pos)))]) if len(pos) else 0.5


def rates(y, p, t, melanoma=None):
    pred = p >= t
    r = {"sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean())}
    if melanoma is not None and melanoma.any():
        r["melanoma_recall"] = float(pred[melanoma].mean())
    return r


def selection_sets(isic_dir, midas_dir):
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    v2 = v2[v2.split == "val"]
    m = pd.read_csv(ROOT / "splits/midas_v1.csv")
    m = m[(m.split == "dev_val") & m.modality.str.startswith("phone")]
    return {"internal_val": (v2.label.astype(int).to_numpy(), [str(Path(isic_dir) / f"{i}.jpg") for i in v2.isic_id],
                             v2.diagnosis_3.fillna("").str.contains("Melanoma").to_numpy()),
            "midas_dev_val": (m.label.astype(int).to_numpy(), [str(Path(midas_dir) / f"{i}.jpg") for i in m.image_id],
                              m.midas_path.str.contains("melanoma").to_numpy())}


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, tf, kind = load_net(opts.run)
    net.to(device).eval()
    sets = selection_sets(opts.isic_dir, opts.midas_dir)
    res = {"run": str(opts.run), "model_kind": kind}
    probs = {}
    for name, (y, paths, mel) in sets.items():
        for tta in (False, True):
            p = p_malignant(net, kind, tf, paths, device, tta)
            res[f"{name}_auroc{'_tta' if tta else ''}"] = float(roc_auc_score(y, p))
            if tta:
                probs[name] = (y, p, mel, p_malignant.last_mel)
    res["S"] = (res["internal_val_auroc_tta"] + res["midas_dev_val_auroc_tta"]) / 2
    res["S_no_tta"] = (res["internal_val_auroc"] + res["midas_dev_val_auroc"]) / 2
    y, p, mel, pm = probs["internal_val"]
    t = threshold_for_sensitivity(y, p)
    res["threshold_internal_val_sens90"] = t
    res["internal_val_at_t"] = rates(y, p, t, mel)
    tm = melanoma_mode(y, p, pm, mel, t) if pm is not None else None
    if tm is not None:
        res["melanoma_mode_threshold"] = tm
        res["melanoma_mode"] = {}
        for name, (yy, pp, mm, pmm) in probs.items():
            pred = (pp >= t) | (pmm >= tm)
            res["melanoma_mode"][name] = {"sensitivity": float(pred[yy == 1].mean()), "specificity": float((~pred[yy == 0]).mean()),
                                          "melanoma_recall": float(pred[mm].mean())}
    y, p, mel, _ = probs["midas_dev_val"]
    res["midas_dev_val_at_t"] = rates(y, p, t, mel)
    Path(opts.run, "selection_score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder (with config.json + checkpoint_best_val.pth.tar)")
    ap.add_argument("--isic_dir", default=str(UNC / "datasets_768/ISIC_clinical"))
    ap.add_argument("--midas_dir", default=str(ROOT / "data/midas/files_768"))
    main(ap.parse_args())
