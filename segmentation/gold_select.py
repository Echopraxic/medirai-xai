"""
Pick the clinical close-ups to hand-annotate as the gold mask set (no ground-truth masks exist for clinical photos).

Drawn from the test split of isic_clinical_v2 only (MSKCC excluded), stratified by source x current mask flag, and balanced on label inside each cell,
so the gold Dice can be reported per source and the ok/review/fail flag can be checked against real errors.
Each image is assigned to a `tune` or `report` half (alternating within cells): tune mask methods and flag
thresholds on `tune`, report on `report`.

    python segmentation/gold_select.py            # writes segmentation/gold/gold_set.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"MILK study team": "MILK", "Memorial Sloan Kettering Cancer Center": "MSKCC"}
QUOTA = {"MILK": 55, "UFES": 45}  # per source, split evenly across the three flags


def short_source(attribution: str) -> str:
    if attribution in SOURCES:
        return SOURCES[attribution]
    return "UFES" if "UFES" in str(attribution) else "other"


def select(qa: pd.DataFrame, seed: int) -> pd.DataFrame:
    v2 = pd.read_csv(ROOT / "splits/isic_clinical_v2.csv")[["isic_id", "split"]]
    qa = qa.drop(columns="split").merge(v2, on="isic_id")
    qa = qa[qa.split == "test"].copy()
    qa["source"] = qa.attribution.map(short_source)
    picks = []
    for source, total in QUOTA.items():
        flags = ["ok", "review", "fail"]
        per_flag = [total // 3 + (k < total % 3) for k in range(3)]
        for flag, n in zip(flags, per_flag):
            cell = qa[(qa.source == source) & (qa.mask_quality == flag)]
            # balance label inside the cell where both labels exist
            parts = [g.sample(frac=1, random_state=seed) for _, g in cell.groupby("label")]
            order = pd.concat([p.iloc[[k]] for k in range(max(map(len, parts), default=0))
                               for p in parts if k < len(p)]) if parts else cell
            picks.append(order.head(n))
    gold = pd.concat(picks).reset_index(drop=True)
    gold["half"] = ["tune" if k % 2 == 0 else "report" for k in range(len(gold))]
    return gold.sample(frac=1, random_state=seed).reset_index(drop=True)  # annotate in random order


def main(opts):
    qa = pd.read_csv(ROOT / "segmentation/output/qa/mask_quality.csv")
    gold = select(qa, opts.seed)
    out = ROOT / "segmentation/gold"
    out.mkdir(parents=True, exist_ok=True)
    gold[["isic_id", "source", "label", "mask_quality", "unet_sam_dice", "half"]].to_csv(out / "gold_set.csv", index=False)
    print(pd.crosstab([gold.source, gold.mask_quality], gold.label, margins=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
