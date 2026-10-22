"""
Final-model report (classifier sprint stage C; prereg 2026-10-21). One chosen run -> calibration, both operating modes
and the K1-K9 numbers that can be computed on the internal test split and the MIDAS dev_test phone photos. Locked sets
are NOT read here (use eval/external_test.py --final once per final candidate).

Fit on validation data only (split v2 val + MIDAS dev_val phone photos where noted):
  temperature T            minimises NLL of the binary logits on split v2 val (8-view TTA logits)
  standard threshold       t: malignant sensitivity 0.90 on split v2 val (calibrated probabilities)
  melanoma mode            malignant if p >= t OR p(melanoma) >= t_m; t_m = highest-specificity value with melanoma
                           recall >= 0.65 on split v2 val (needs a run trained with --mel_head)
Reported on split v2 test and MIDAS dev_test (phone):
  AUROC, ECE (10 bins, top-label), sensitivity / specificity / melanoma recall in both modes, deferral curve
  (accuracy retained at 20/30% deferral by entropy), per-source AUROC (MILK, UFES).

    python eval/final_model_report.py --run <run folder under uncertaintyNet-main/output> --out eval/output/stage_c/final
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "classifier"))

from preprocess_ablation import DIHEDRAL  # noqa: E402
from selection_score import load_net, threshold_for_sensitivity  # noqa: E402
from PIL import Image  # noqa: E402


def sets(isic_dir, midas_dir):
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv").set_index("isic_id")
    mid = pd.read_csv(ROOT / "splits/midas_v1.csv")
    mid = mid[mid.modality.str.startswith("phone")].set_index("image_id")
    out = {}
    for name, df, d in (("v2_val", v2[v2.split == "val"], isic_dir), ("v2_test", v2[v2.split == "test"], isic_dir)):
        out[name] = {"ids": df.index.tolist(), "dir": d, "label": df.label.astype(int).to_numpy(),
                     "melanoma": df.diagnosis_3.fillna("").str.contains("Melanoma").to_numpy(), "source": df.source.to_numpy()}
    for name, split in (("midas_dev_val", "dev_val"), ("midas_dev_test", "dev_test")):
        m = mid[mid.split == split]
        out[name] = {"ids": m.index.tolist(), "dir": midas_dir, "label": m.label.astype(int).to_numpy(),
                     "melanoma": m.midas_path.str.contains("melanoma").to_numpy(), "source": np.array(["MIDAS"] * len(m))}
    return out


@torch.no_grad()
def tta_logits(net, kind, tf, ids, folder, device, batch_size=32):
    """Mean over 8 views of the binary logit margin (logit_1 - logit_0) and of p(melanoma) when the head exists."""
    margins, mels = [], []
    has_mel = getattr(net, "mel_head", None) is not None
    for i in range(0, len(ids), batch_size):
        x = torch.stack([tf(Image.open(Path(folder) / f"{k}.jpg").convert("RGB")) for k in ids[i:i + batch_size]]).to(device)
        m, pm = [], []
        for v in DIHEDRAL:
            logits, f = net(v(x))[:2] if kind == "sprint" else (net(v(x)), None)
            logits = logits.float()
            m.append(logits[:, 1] - logits[:, 0])
            if has_mel:
                pm.append(torch.sigmoid(net.mel_head(f.float()).squeeze(1)))
        margins.append(torch.stack(m).mean(0).cpu().numpy())
        if has_mel:
            mels.append(torch.stack(pm).mean(0).cpu().numpy())
    return np.concatenate(margins), (np.concatenate(mels) if mels else None)


def ece(p, y, bins=10):
    conf, pred = np.maximum(p, 1 - p), (p >= 0.5).astype(int)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def fit_temperature(margin, y):
    def nll(T):
        z = margin / T
        return float(np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))
    return float(minimize_scalar(nll, bounds=(0.3, 10), method="bounded").x)


def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def operating(p, pm, d, t, tm):
    y, mel = d["label"], d["melanoma"]
    out = {}
    for mode, pred in (("standard", p >= t), ("melanoma_sensitive", (p >= t) | (pm >= tm) if pm is not None else None)):
        if pred is None:
            continue
        out[mode] = {"sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
                     "melanoma_recall": float(pred[mel].mean()) if mel.any() else None, "flagged_share": float(pred.mean())}
    return out


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, tf, kind = load_net(UNC / "output" / opts.run)
    net.to(device).eval()
    data = sets(opts.isic_dir, opts.midas_dir)
    raw = {n: tta_logits(net, kind, tf, d["ids"], d["dir"], device) for n, d in data.items()}
    T = fit_temperature(raw["v2_val"][0], data["v2_val"]["label"])
    prob = {n: 1 / (1 + np.exp(-raw[n][0] / T)) for n in data}
    t = threshold_for_sensitivity(data["v2_val"]["label"], prob["v2_val"])
    tm = None
    pm_val = raw["v2_val"][1]
    if pm_val is not None:
        y, mel = data["v2_val"]["label"], data["v2_val"]["melanoma"]
        best = None
        for cand in np.quantile(pm_val, np.linspace(0.3, 0.995, 140)):
            pred = (prob["v2_val"] >= t) | (pm_val >= cand)
            if pred[mel].mean() >= 0.65 and (best is None or (~pred[y == 0]).mean() > best[1]):
                best = (float(cand), float((~pred[y == 0]).mean()))
        tm = best[0] if best else None
    res = {"run": opts.run, "temperature": T, "threshold_standard": t, "threshold_melanoma_head": tm,
           "ece_before_scaling_v2_val": ece(1 / (1 + np.exp(-raw["v2_val"][0])), data["v2_val"]["label"])}
    for n in ("v2_val", "v2_test", "midas_dev_val", "midas_dev_test"):
        d, p, pm = data[n], prob[n], raw[n][1]
        r = {"n": len(d["ids"]), "auroc": float(roc_auc_score(d["label"], p)), "ece": ece(p, d["label"]),
             **operating(p, pm, d, t, tm)}
        e = entropy(p)
        wrong = ((p >= 0.5).astype(int) != d["label"])
        for share in (0.2, 0.3):
            keep = e <= np.quantile(e, 1 - share)
            r[f"retained_accuracy_at_{int(share * 100)}pct_deferral"] = float(1 - wrong[keep].mean())
        if n == "v2_test":
            r["auroc_by_source"] = {s: float(roc_auc_score(d["label"][d["source"] == s], p[d["source"] == s]))
                                    for s in ("MILK", "UFES") if len(set(d["label"][d["source"] == s])) == 2}
        res[n] = r
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_model_report.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "scores.npz", **{f"{n}_prob": prob[n] for n in prob},
                        **{f"{n}_pmel": raw[n][1] for n in raw if raw[n][1] is not None})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--isic_dir", default=str(UNC / "datasets_768/ISIC_clinical"))
    ap.add_argument("--midas_dir", default=str(ROOT / "data/midas/files_768"))
    main(ap.parse_args())
