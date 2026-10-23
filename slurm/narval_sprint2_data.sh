#!/bin/bash
#SBATCH --job-name=medirai-sprint2-data
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Prereg 2026-10-23 section 1: build the darker-skin dev set (splits/f17k_dsd_v1.csv), split v7 (v6 + Mendeley pairs,
# train only) and splits/mendeley_pairs_v1.csv, caching the images into the training folder; then the DINOv2-L
# embeddings of the DSD images (scope distance and the darker-skin router read them).
#   sbatch --account=def-omar12_gpu slurm/narval_sprint2_data.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
E=/scratch/mikeymck/medirai/external
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v4}
export HF_HOME=${HF_HOME:-$SCRATCH/hf} HF_HUB_OFFLINE=1
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

python splits/make_split_v7.py --f17k_dir "$E/fitzpatrick17k_full" --mendeley_dir "$E/mendeley_synthetic_fst" \
  --image_dir "$DATA/ISIC_clinical" \
  --locked_dirs "$E/ddi_diverse_dermatology_images/files_768" "$E/ddi2_diverse_dermatology_images_2/files_768" "$E/mcr_sl/MCR-SL_1024" \
  --locked_hashes eval/output/external/_overview_dhash.npz
python classifier/embed_probe.py embed --model dinov2 --split_csv splits/f17k_dsd_v1.csv \
  --image_dir "$DATA/ISIC_clinical" --tag dsd
