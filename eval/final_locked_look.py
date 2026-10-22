"""
The single final look at the four locked external sets for the chosen model (classifier sprint, prereg 2026-10-21).
Every look is appended to eval/output/external/looks.csv through external_test.log_look. The model, temperature and
thresholds come from eval/final_model_report.py (all fit on validation data); nothing here is tuned.

Per set: AUROC with a lesion/patient bootstrap CI, ECE (calibrated), sensitivity / specificity / melanoma recall in both
operating modes, entropy AUROC for errors, Fitzpatrick-group AUROC where tone is known, and a paired comparison against
the v2 seed-0 baseline scored earlier (same images; its predictions are in eval/output/external/<set>/predictions.csv).
K6 (mean AUROC), K7 (worst Fitzpatrick group on DDI), K5 (ECE per set) and the K8 entropy part are computed from these.

    python eval/final_locked_look.py --run <run> --final "final candidate, kdv6_mel seed 1"
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "classifier"))

import external_test as E  # noqa: E402
from final_model_report import ece, entropy  # noqa: E402
from preprocess_ablation import DIHEDRAL  # noqa: E402
from selection_score import load_net  # noqa: E402

BASELINE = "rorqual_v2_lr3e-4_20261014"
LOCKED = ["isic_overview", "mcr_sl", "ddi", "ddi2"]


@torch.no_grad()
def margins(net, kind, tf, images, device, batch_size=32):
    m, pm = [], []
    has_mel = getattr(net, "mel_head", None) is not None
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        mm, pp = [], []
        for v in DIHEDRAL:
            logits, f = net(v(x))[:2]
            logits = logits.float()
            mm.append(logits[:, 1] - logits[:, 0])
            if has_mel:
                pp.append(torch.sigmoid(net.mel_head(f.float()).squeeze(1)))
        m.append(torch.stack(mm).mean(0).cpu().numpy())
        if has_mel:
            pm.append(torch.stack(pp).mean(0).cpu().numpy())
    return np.concatenate(m), (np.concatenate(pm) if pm else None)


def mode_metrics(d, pred):
    y, mel = d.label.to_numpy(), d.diagnosis.str.contains("Melanoma", case=False).to_numpy()
    return {"sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
            "melanoma_recall": float(pred[mel].mean()) if mel.any() else None, "n_melanoma": int(mel.sum())}


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep = json.loads((ROOT / opts.final_report / "final_model_report.json").read_text())
    scores = np.load(ROOT / opts.final_report / "scores.npz")
    T, t, tm = rep["temperature"], rep["threshold_standard"], rep["threshold_melanoma_head"]
    eoe_cut = float(np.quantile(entropy(scores["v2_val_prob"]), 0.75))
    net, tf, kind = load_net(UNC / "output" / opts.run)
    net.to(device).eval()
    ref_hashes = E.v2_hashes(ROOT / "eval/output/external/_v2_dhash.npz")
    out_dir = ROOT / "eval/output/stage_c/locked_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    final, tag = {}, "final"
    sources = opts.sources.split(",") if opts.sources else LOCKED
    for src in sources:
        if src in E.LOCKED:
            print(f"locked set {src}: look #{E.log_look(Namespace(source=src, runs=[opts.run], final=opts.final))}", flush=True)
        df = E.SOURCES[src]().reset_index(drop=True)
        images = [E.load_768(p) for p in df.path]
        dup = E.near_duplicates([E.dhash(im) for im in images], *ref_hashes)
        keep = np.array([d is None for d in dup])
        df, images = df[keep].reset_index(drop=True), [im for im, k in zip(images, keep) if k]
        margin, pmel = margins(net, kind, tf, images, device)
        p = 1 / (1 + np.exp(-margin / T))
        d = df.assign(p_malignant=p, pred=(p >= t).astype(int), entropy_of_expected=entropy(p))
        summary = E.summarize(d, eoe_cut, opts.n_boot)
        summary["ece_calibrated"] = ece(p, d.label.to_numpy())
        modes = {"standard": mode_metrics(d, p >= t)}
        if pmel is not None and tm is not None:
            modes["melanoma_sensitive"] = mode_metrics(d, (p >= t) | (pmel >= tm))
        summary["operating_modes"] = modes
        base = pd.read_csv(ROOT / "eval/output/external" / ("midas_dev_val" if src == "midas" else src) / "predictions.csv")
        assert len(base) == len(d) and (base.image_id.to_numpy() == d.image_id.to_numpy()).all(), "image order mismatch"
        pair = base.copy()
        pair[f"{tag}__p_malignant"], pair[f"{tag}__pred"] = d.p_malignant.to_numpy(), d.pred.to_numpy()
        summary["paired_vs_v2_seed0"] = E.paired_diff(pair.assign(group=d.group.to_numpy()), BASELINE, tag, opts.n_boot)
        summary["baseline_v2_seed0_auroc"] = float(roc_auc_score(d.label, base[f"{BASELINE}__p_malignant"]))
        final[src] = summary
        d.drop(columns=["path"]).to_csv(out_dir / f"{opts.out_name}_{src}_predictions.csv", index=False)
        print(src, "AUROC", round(summary["AUROC"]["value"], 3), "vs v2 s0", round(summary["baseline_v2_seed0_auroc"], 3), flush=True)
    k = {"K6_mean_auroc": float(np.mean([final[s]["AUROC"]["value"] for s in sources])),
         "K6_baseline_v2_seed0": float(np.mean([final[s]["baseline_v2_seed0_auroc"] for s in sources])),
         "K5_ece_by_set": {s: final[s]["ece_calibrated"] for s in sources},
         "K7_ddi_auroc_by_tone": {g: v["AUROC"] for g, v in final.get("ddi", {}).get("by tone", {}).items()},
         "K8_entropy_auroc_for_errors_mean": float(np.mean([final[s]["entropy AUROC for errors"] for s in sources]))}
    (out_dir / (opts.out_name + ".json")).write_text(json.dumps({"run": opts.run, "temperature": T, "threshold": t,
                                                           "melanoma_threshold": tm, "sets": final, "k": k}, indent=1))
    print(json.dumps(k, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--final", required=True, help="why this look is needed (logged)")
    ap.add_argument("--final_report", default="eval/output/stage_c/final")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--sources", default="", help="comma list (default: the four locked sets); midas = dev_val dry run")
    ap.add_argument("--out_name", default="locked_final")
    main(ap.parse_args())
