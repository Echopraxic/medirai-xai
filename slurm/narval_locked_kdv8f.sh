#!/bin/bash
#SBATCH --job-name=medirai-locked-kdv8f
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Final-model locked look on the sprint part 3 model (kdv8f_mel seed 3; note 2026-11-06 section 6). Two steps:
#   1. eval/final_model_report.py: temperature and both thresholds fit on split v2 val (no locked set read)
#   2. eval/final_locked_look.py: one look at isic_overview, mcr_sl, ddi, ddi2 (each logged to looks.csv), paired vs
#      v2 seed 0; then DDI AUROC by Fitzpatrick group next to kdv6_mel s1 (eval/output/stage_c/locked_final.json)
# kdv6_mel's report and look stay where they are; this run writes beside them.
#   sbatch --account=def-omar12_gpu slurm/narval_locked_kdv8f.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
RUN=${RUN:-narval_stageb_kdv8f_mel_s3_20261030/run}
REPORT=eval/output/stage_c/final_kdv8f_mel_s3
NAME=locked_final_kdv8f_mel_s3
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

python eval/final_model_report.py --run "$RUN" --out "$REPORT"
python eval/final_locked_look.py --run "$RUN" --final_report "$REPORT" --out_name "$NAME" \
  --final "final model after sprint part 3 (F1 adopted 2026-10-30): kdv8f_mel seed 3"

python - <<EOF
import json
d = "eval/output/stage_c/locked_final"
old = json.load(open(f"{d}/locked_final.json"))["k"]
new = json.load(open(f"{d}/$NAME.json"))["k"]
print("K6 mean AUROC  kdv6_mel s1", round(old["K6_mean_auroc"], 3), " kdv8f_mel s3", round(new["K6_mean_auroc"], 3))
for g in new["K7_ddi_auroc_by_tone"]:
    print("DDI", g, " kdv6_mel s1", old["K7_ddi_auroc_by_tone"].get(g), " kdv8f_mel s3", new["K7_ddi_auroc_by_tone"][g])
EOF

# Darkness contrast on DDI from the predictions above (descriptive; the DDI read is logged), and on the darker-skin
# dev set for the same model (selection data)
python eval/darkness_contrast.py ddi --features features/output/external/ddi_features_v1.0.parquet \
  --predictions "eval/output/stage_c/locked_final/${NAME}_ddi_predictions.csv" \
  --final "darkness contrast on DDI, descriptive (note 2026-11-06 item 2)" \
  --out eval/output/darkness_contrast/ddi_kdv8f_mel_s3.json
python eval/darkness_contrast.py dsd --features features/output/external/dsd_features_v1.0.parquet \
  --tta "uncertaintyNet-main/output/$RUN/tta_scores.npz" --out eval/output/darkness_contrast/dsd_kdv8f_mel_s3.json

# F1 trains on Fitzpatrick17k-C FST I-IV: are the dev set's FST IV comparison images in its training split?
python - <<'EOF'
import json, pandas as pd
cfg = json.load(open("uncertaintyNet-main/output/narval_stageb_kdv8f_mel_s3_20261030/run/config.json"))
split = pd.read_csv(f"splits/{cfg['split_file']}")
dsd = pd.read_parquet("features/output/external/dsd_features_v1.0.parquet")
iv = set(dsd.loc[~dsd.dsd, "image_id"])
train = set(split.loc[split.split == "train", "isic_id"])
print(f"DSD FST IV images in kdv8f training split: {len(iv & train)} of {len(iv)}")
EOF
