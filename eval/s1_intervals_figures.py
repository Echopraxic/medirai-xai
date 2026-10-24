"""
S1-9 intervals and figures for the final candidate on the locked sets (descriptive; nothing is selected).

Input: all_strata.csv from eval/s1_error_lists.py. Images are resampled by lesion (`group` column) with a cluster
bootstrap (2,000 resamples, seed 0) and every rate gets a 95% percentile interval. Rates are computed per set and,
where a tone column exists (DDI, DDI-2), per Fitzpatrick group.

Outputs (in --out; derived from DDI/DDI-2 rows, keep local):
  intervals.csv / intervals.md   n, n_malignant and each metric with its interval
  fig_sensitivity_by_tone.png    DDI sensitivity and specificity by Fitzpatrick group, with intervals
  fig_deferral_curves.png        retained accuracy vs share deferred by entropy, one line per set

    python eval/s1_intervals_figures.py --strata eval/output/s1_error_lists/all_strata.csv \
        --out eval/output/s1_error_lists/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

N_BOOT = 2000
SEED = 0
SETS = ["isic_overview", "mcr_sl", "ddi", "ddi2"]
LABELS = {"isic_overview": "ISIC overview", "mcr_sl": "MCR-SL", "ddi": "DDI", "ddi2": "DDI-2"}
# Categorical slots 1-4 of the dataviz reference palette (validated light mode; lines are direct-labelled)
COLORS = {"isic_overview": "#2a78d6", "mcr_sl": "#eb6834", "ddi": "#1baf7a", "ddi2": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def rates(d):
    mal, ben, mel = d.label == 1, d.label == 0, d.melanoma.astype(bool)
    wrong = d.pred != d.label
    out = {
        "sensitivity": (d.pred[mal] == 1).mean() if mal.any() else np.nan,
        "specificity": (d.pred[ben] == 0).mean() if ben.any() else np.nan,
        "melanoma recall": (d.pred[mel] == 1).mean() if mel.sum() >= 5 else np.nan,
        "share deferred": d.deferred.mean(),
        "error rate, deferred": wrong[d.deferred].mean() if d.deferred.any() else np.nan,
        "error rate, retained": wrong[~d.deferred].mean() if (~d.deferred).any() else np.nan,
    }
    out["entropy AUROC for errors"] = (roc_auc_score(wrong, d.entropy_of_expected)
                                       if 0 < wrong.sum() < len(d) else np.nan)
    return out


def cluster_bootstrap(d, n_boot=N_BOOT, seed=SEED):
    """Point estimate and 95% percentile interval per metric, resampling whole lesions."""
    point = rates(d)
    groups = d.groupby("group").indices
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[k]] for k in pick])
        draws.append(rates(d.iloc[idx]))
    draws = pd.DataFrame(draws)
    return {m: (v, *np.nanpercentile(draws[m], [2.5, 97.5])) if np.isfinite(v) else (np.nan, np.nan, np.nan)
            for m, v in point.items()}


def interval_table(strata, n_boot=N_BOOT):
    rows = []
    units = [(s, "all", strata[strata.set == s]) for s in SETS]
    for s in SETS:
        d = strata[strata.set == s]
        if (d.tone != "not recorded").any():
            units += [(s, t, g) for t, g in d.groupby("tone")]
    for s, tone, d in units:
        r = {"set": s, "tone": tone, "n": len(d), "n_malignant": int(d.label.sum()),
             "n_melanoma": int(d.melanoma.sum())}
        for m, (v, lo, hi) in cluster_bootstrap(d.reset_index(drop=True), n_boot).items():
            r[m], r[f"{m} lo"], r[f"{m} hi"] = v, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(t):
    show = t[["set", "tone", "n", "n_malignant", "n_melanoma"]].copy()
    for m in ["sensitivity", "specificity", "melanoma recall", "share deferred", "error rate, deferred",
              "error rate, retained", "entropy AUROC for errors"]:
        show[m] = [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(v) else "–"
                   for v, lo, hi in zip(t[m], t[f"{m} lo"], t[f"{m} hi"])]
    return show


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_by_tone(t, out):
    d = t[(t.set == "ddi") & (t.tone != "all")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, m in zip(axes, ["sensitivity", "specificity"]):
        x = np.arange(len(d))
        ax.errorbar(x, d[m], yerr=[d[m] - d[f"{m} lo"], d[f"{m} hi"] - d[m]], fmt="o", color=COLORS["isic_overview"],
                    ecolor=COLORS["isic_overview"], elinewidth=2, capsize=0, markersize=8)
        for xi, v, n in zip(x, d[m], d.n_malignant if m == "sensitivity" else d.n - d.n_malignant):
            ax.annotate(f"{v:.2f}\n(n={n})", (xi, v), xytext=(10, -4), textcoords="offset points", fontsize=8,
                        color=INK)
        ax.set_xticks(x, [f"FST {s}" for s in d.tone])
        ax.set_xlim(-0.5, len(d) - 0.2)
        ax.set_ylim(0, 1)
        ax.set_title(f"DDI {m}", fontsize=10, color=INK, loc="left")
        style(ax)
    fig.text(0.01, 0.01, "Final candidate, standard threshold; 95% lesion-level bootstrap intervals.", fontsize=8,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_deferral(strata, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    shares = np.linspace(0, 0.9, 46)
    ends = {}
    for s in SETS:
        d = strata[strata.set == s]
        order = d.entropy_of_expected.sort_values(ascending=False).index
        correct = (d.pred == d.label).loc[order].to_numpy()
        acc = [correct[int(round(q * len(d))):].mean() for q in shares]
        ax.plot(shares * 100, acc, color=COLORS[s], linewidth=2)
        ends[s] = acc[-1]
    # Direct labels at the line ends, nudged apart when they would collide
    placed = []
    for s in sorted(ends, key=ends.get):
        y = ends[s]
        if placed and y - placed[-1] < 0.02:
            y = placed[-1] + 0.02
        placed.append(y)
        ax.annotate(LABELS[s], (shares[-1] * 100, y), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=INK, va="center")
    ax.axvline(30, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("K8: 30% deferral", (30, ax.get_ylim()[0]), xytext=(4, 4), textcoords="offset points",
                fontsize=8, color=MUTED)
    ax.set_xlabel("Share of images deferred, highest entropy first (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("Accuracy on retained images", fontsize=9, color=MUTED)
    ax.set_xlim(0, 105)
    ax.set_title("Entropy deferral helps on ISIC overview and MCR-SL, not on DDI or DDI-2", fontsize=10,
                 color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(opts):
    strata = pd.read_csv(opts.strata)
    strata["tone"] = strata.tone.fillna("not recorded")
    out = Path(opts.out)
    out.mkdir(parents=True, exist_ok=True)
    t = interval_table(strata, opts.n_boot)
    t.round(4).to_csv(out / "intervals.csv", index=False)
    md = "\n".join(["# S1-9 rates with 95% lesion-level bootstrap intervals", "",
                    f"{opts.n_boot} resamples, seed {SEED}. Melanoma recall only with >= 5 melanomas.", "",
                    fmt(t).to_markdown(index=False), ""])
    (out / "intervals.md").write_text(md)
    fig_by_tone(t, out / "fig_sensitivity_by_tone.png")
    fig_deferral(strata, out / "fig_deferral_curves.png")
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strata", default="eval/output/s1_error_lists/all_strata.csv")
    ap.add_argument("--out", default="eval/output/s1_error_lists/figures")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    main(ap.parse_args())
