#!/bin/bash
#SBATCH --job-name=medirai-sprint3-data
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Prereg 2026-10-30 section 1: DSD v2 (Fitzpatrick17k-C cleaned), split v8f (+ F17k-C FST I-IV), v8s (+ SD-198) and
# the six learning-curve splits, caching the new images into the training folder. CPU only.
#   sbatch --account=def-omar12 slurm/narval_sprint3_data.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
E=/scratch/mikeymck/medirai/external
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v4}
module load python/3.12
source "$REPO/.venv/bin/activate"
cd "$REPO"

python splits/make_split_v8.py --f17k_dir "$E/fitzpatrick17k_full" --f17k_c "$E/fitzpatrick17k_full/Fitzpatrick17k-C.csv" \
  --sd198_dir "$E/sd198" --image_dir "$DATA/ISIC_clinical" \
  --locked_dirs "$E/ddi_diverse_dermatology_images/files_768" "$E/ddi2_diverse_dermatology_images_2/files_768" "$E/mcr_sl/MCR-SL_1024" \
  --locked_hashes eval/output/external/_overview_dhash.npz
