"""
Stage C test (prereg 2026-10-21; plan §6 fallback): can several trained models be combined into one better model?
Scored on the SELECTION data only (split v2 val + MIDAS dev_val phone photos, 8-view TTA).

For every --group name=run1,run2,...:
  single_mean   mean selection score of the member models (the reference for the two combinations)
  ensemble      mean of the members' probabilities (needs K forward passes at inference)
  soup          uniform average of the members' weights = ONE ResNet50 (BatchNorm statistics recomputed on 2,000
                training images after averaging; members must share an architecture)
plus `--all_ensemble`: probability ensemble across every member of every group.

    python eval/soup_ensemble.py --group ref=A,B,C --group hyg=D,E,F --out eval/output/stage_c/soup_ensemble.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UNC = ROOT / "uncertaintyNet-main"
sys.path.insert(0, str(UNC))
sys.path.insert(0, str(ROOT / "eval"))

from selection_score import load_net, p_malignant, rates, selection_sets, threshold_for_sensitivity  # noqa: E402


def probs_of(net, kind, tf, sets, device):
    return {name: p_malignant(net, kind, tf, paths, device, tta=True) for name, (_, paths, _) in sets.items()}


def score(probs, sets):
    r = {f"{n}_auroc": float(roc_auc_score(sets[n][0], probs[n])) for n in sets}
    r["S"] = (r["internal_val_auroc"] + r["midas_dev_val_auroc"]) / 2
    y, mel = sets["internal_val"][0], sets["internal_val"][2]
    t = threshold_for_sensitivity(y, probs["internal_val"])
    r["internal_val_at_t"] = rates(y, probs["internal_val"], t, mel)
    r["midas_dev_val_at_t"] = rates(sets["midas_dev_val"][0], probs["midas_dev_val"], t, sets["midas_dev_val"][2])
    return r


def soup(nets, tf, device, isic_dir, n_bn=2000):
    """Uniform weight average; BatchNorm running statistics recomputed on training images."""
    model = copy.deepcopy(nets[0])
    states = [n.state_dict() for n in nets]
    avg = {}
    for k, v in states[0].items():
        avg[k] = (torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v)
    model.load_state_dict(avg)
    train = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")
    ids = train[train.split == "train"].isic_id.sample(n_bn, random_state=0).tolist()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                     # cumulative average over the BN-recompute pass
    model.to(device).train()
    fwd = (lambda x: model(x)[0]) if hasattr(model, "feat_dim") else model
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            x = torch.stack([tf(Image.open(Path(isic_dir) / f"{k}.jpg").convert("RGB")) for k in ids[i:i + 64]]).to(device)
            fwd(x)
    return model.eval()


def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = selection_sets(opts.isic_dir, opts.midas_dir)
    if opts.limit:  # smoke test: a class-balanced subsample of each set
        for name, (y, paths, mel) in list(sets.items()):
            pick = np.concatenate([np.flatnonzero(y == c)[:opts.limit // 2] for c in (0, 1)])
            sets[name] = (y[pick], [paths[i] for i in pick], mel[pick])
    res, all_probs = {}, []
    for spec in opts.group:
        name, runs = spec.split("=")
        runs = runs.split(",")
        nets, member_probs, member_scores = [], [], []
        for run in runs:
            net, tf, kind = load_net(UNC / "output" / run)
            net.to(device).eval()
            p = probs_of(net, kind, tf, sets, device)
            member_probs.append(p)
            member_scores.append(score(p, sets))
            nets.append(net)
        ens = {n: np.mean([p[n] for p in member_probs], axis=0) for n in sets}
        sp = soup([n.cpu() for n in nets], tf, device, opts.isic_dir)
        res[name] = {"members": runs,
                     "single_mean": {k: float(np.mean([m[k] for m in member_scores])) for k in ("internal_val_auroc", "midas_dev_val_auroc", "S")},
                     "ensemble": score(ens, sets),
                     "soup": score(probs_of(sp, kind, tf, sets, device), sets)}
        all_probs += member_probs
        print(name, json.dumps({k: (round(v["S"], 4) if isinstance(v, dict) and "S" in v else None)
                                for k, v in res[name].items() if k != "members"}), flush=True)
    if opts.all_ensemble and len(all_probs) > 1:
        res["all_groups_ensemble"] = score({n: np.mean([p[n] for p in all_probs], axis=0) for n in sets}, sets)
        print("all groups ensemble S", round(res["all_groups_ensemble"]["S"], 4))
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="name=run1,run2,... (run folders under uncertaintyNet-main/output)")
    ap.add_argument("--all_ensemble", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: images per set")
    ap.add_argument("--isic_dir", default=str(UNC / "datasets_768/ISIC_clinical"))
    ap.add_argument("--midas_dir", default=str(ROOT / "data/midas/files_768"))
    main(ap.parse_args())
