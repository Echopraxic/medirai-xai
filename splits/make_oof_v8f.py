"""
Out-of-fold split files for the C1 surrogate (plan decision 2026-11-04): split v8f train cut into 5 folds grouped by
group_id (lesion/patient) and stratified by label, seed 0. File k = split v8f with fold k's train rows marked split="oof":
classifier/train_sprint.py trains on split == "train" only, and eval/export_predictions.py predicts every row, so each
fold model gives honest predictions for the rows it never saw. Val and test rows are unchanged.

    python splits/make_oof_v8f.py   -> splits/isic_v8f_oof_k{0..4}.csv (gitignored: contain MIDAS rows), oof_folds_v8f.csv
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
SRC = "isic_v8f_v6_plus_f17k.csv"
K, SEED = 5, 0


def folds(df):
    """isic_id -> fold for the train rows."""
    tr = df[df.split == "train"].reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
    out = pd.Series(-1, index=tr.index)
    for k, (_, te) in enumerate(cv.split(tr, tr.label.astype(int), tr.group_id.astype(str))):
        out[te] = k
    return dict(zip(tr.isic_id, out))


def main():
    df = pd.read_csv(HERE / SRC)
    f = folds(df)
    pd.DataFrame({"isic_id": list(f), "fold": list(f.values())}).to_csv(HERE / "oof_folds_v8f.csv", index=False)
    for k in range(K):
        d = df.copy()
        d.loc[d.isic_id.map(f).eq(k), "split"] = "oof"
        d.to_csv(HERE / f"isic_v8f_oof_k{k}.csv", index=False)
        print(k, d.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
