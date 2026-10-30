#!/bin/bash
#SBATCH --job-name=medirai-scin-embed
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# SCIN: resize every image, label table + case-group split (interpret/scin_labels.py), DINOv2-L embeddings of all images,
# then the split again with near-duplicate joining (--emb). For the tone-detector / phone-router refit.
#   sbatch --account=def-omar12_gpu slurm/narval_scin_embed.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
E=/scratch/mikeymck/medirai/external/scin
export HF_HOME=${HF_HOME:-$SCRATCH/hf} HF_HUB_OFFLINE=1
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

python tools/resize_scin.py --scin_dir "$E" --out "$E/files_768"
python interpret/scin_labels.py --data "$E" --out splits/scin_v1.csv
python classifier/embed_probe.py embed --model dinov2 --split_csv splits/scin_v1.csv --id_column image_id \
  --image_dir "$E/files_768" --tag scin
python interpret/scin_labels.py --data "$E" --out splits/scin_v1.csv --emb classifier/output/embeddings/dinov2__scin.npz
