"""
Classifier sprint part 3 (prereg S2_CLASSIFIER_PREREG_2026-10-30.md): arm adoption on DSD v2 and the darker-skin
learning curve, from each run's tta_scores.npz (eval/tta_scores.py). Reads no locked set.

  arms   S0 (kdv6_mel s0-4), F1 (kdv8f_mel), D1 (kdv8s_mel): the part-2 rule (eval/sprint2_analysis.arms) with the
         darker-skin AUROC on DSD v2 rows only (splits/f17k_dsd_v2.csv)
  curve  held-out fold k of DSD v2: n = 0 (F1 seeds 0-2), 1, 2 other folds added (kdv8f_lc, seeds 0-2); AUROC of the
         seed-mean probability per fold, averaged over folds; gain n=2 - n=0 and the slope per doubling of
         (1 + images added), with 95% intervals from resampling duplicate groups within each held-out fold

    python eval/sprint3_analysis.py   ->  eval/output/sprint3/analysis.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from sprint2_analysis import N_BOOT, UNC, arms, auroc, load_run, meta, sig  # noqa: E402

ARMS = {"S0": "kdv6_mel", "F1": "kdv8f_mel", "D1": "kdv8s_mel"}
READING_BAR = 0.02   # prereg section 3: a gain < 0.02 at n = 2 means more atlas data will not reach K7


def find_runs(pattern):
    """Run folders (relative to uncertaintyNet-main/output) matching narval_stageb_<pattern>_s<seed>_<date>, by seed."""
    out = {}
    for p in sorted((UNC / "output").glob("narval_stageb_*/run/tta_scores.npz")):
        m = re.fullmatch(rf"narval_stageb_{pattern}_s(\d+)_\d{{8}}", p.parent.parent.name)
        if m:
            out[int(m.group(1))] = p.parent.relative_to(UNC / "output").as_posix()
    return out


def dsd_v2_meta(M, dsd2):
    """Point the part-2 'dsd' mask at the DSD v2 rows, so selection() scores the cleaned dev set."""
    M = dict(M)
    M["dsd"] = M["dsd"].assign(dsd=M["dsd"].index.isin(dsd2.isic_id))
    return M


def fold_auroc_boot(y, p_by_n, groups, seed=0):
    """Per held-out fold: AUROC per n on group resamples (same resample for every n)."""
    rng = np.random.default_rng(seed)
    g = pd.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in g}
    out = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[k] for k in rng.choice(g, len(g))])
        out.append({n: auroc(y[take], p[take]) for n, p in p_by_n.items()})
    return out


def slope(added, aurocs):
    x = np.log2(1 + np.asarray(added, float))
    return float(np.polyfit(x, np.asarray(aurocs, float), 1)[0])


def curve(dsd2):
    f1 = find_runs(ARMS["F1"])
    res, boots, added = {"folds": {}}, [], {0: [], 1: [], 2: []}
    for k in range(3):
        held = dsd2[dsd2.fold == k]
        runs = {0: [f1[s] for s in (0, 1, 2) if s in f1]}
        for n in (1, 2):
            lc = find_runs(f"kdv8f_lc_k{k}_n{n}")
            runs[n] = [lc[s] for s in (0, 1, 2) if s in lc]
        if any(len(r) < 3 for r in runs.values()):
            res["folds"][k] = {"missing_runs": {n: 3 - len(r) for n, r in runs.items()}}
            continue
        y = held.label.astype(int).values
        p = {n: np.mean([sig(load_run(r)["dsd"].margin.loc[held.isic_id].values) for r in rs], axis=0) for n, rs in runs.items()}
        for n in (0, 1, 2):
            added[n].append(int(dsd2[dsd2.fold.isin([(k + i) % 3 for i in range(1, n + 1)])].shape[0]))
        res["folds"][k] = {"n_held_out": len(held), **{f"auroc_n{n}": auroc(y, p[n]) for n in p}}
        boots.append(fold_auroc_boot(y, p, held.group_id.values, seed=k))
    if len(boots) < 3:
        res["status"] = "incomplete"
        return res
    mean_auc = {n: float(np.mean([res["folds"][k][f"auroc_n{n}"] for k in range(3)])) for n in (0, 1, 2)}
    mean_added = {n: float(np.mean(added[n])) for n in (0, 1, 2)}
    b_auc = [{n: np.mean([boots[k][i][n] for k in range(3)]) for n in (0, 1, 2)} for i in range(N_BOOT)]
    gains = [b[2] - b[0] for b in b_auc]
    slopes = [slope(list(mean_added.values()), [b[n] for n in (0, 1, 2)]) for b in b_auc]
    gain = mean_auc[2] - mean_auc[0]
    res.update({"auroc_mean_over_folds": mean_auc, "images_added_mean": mean_added,
                "gain_n2_minus_n0": gain, "gain_ci95": [float(np.nanpercentile(gains, 2.5)), float(np.nanpercentile(gains, 97.5))],
                "slope_per_doubling": slope(list(mean_added.values()), list(mean_auc.values())),
                "slope_ci95": [float(np.nanpercentile(slopes, 2.5)), float(np.nanpercentile(slopes, 97.5))],
                "reading": ("more atlas darker-skin data will not reach K7 (gain < 0.02)" if gain < READING_BAR
                            else "gain >= 0.02: extrapolate the slope for the T1 target (with its interval)")})
    return res


def main():
    dsd2 = pd.read_csv(ROOT / "splits/f17k_dsd_v2.csv")
    M = dsd_v2_meta(meta(), dsd2)
    found = {a: find_runs(p) for a, p in ARMS.items()}
    runs = {a: [r[s] for s in range(5) if s in r] for a, r in found.items()}
    out = {"runs_found": {a: len(r) for a, r in runs.items()}}
    if all(len(r) == 5 for r in runs.values()):
        out["arms"] = arms(runs, M)
    else:
        out["arms"] = "incomplete: need 5 seeds per arm"
    out["learning_curve"] = curve(dsd2)
    path = ROOT / "eval/output/sprint3/analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
