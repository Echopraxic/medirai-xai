#!/bin/bash
# 7-class diagnosis head ResNet50 on split v2 (classifier/train_multiclass.py), then the binary baseline report.
#
#   sbatch --account=def-omar12_gpu slurm/narval_multiclass.sh
#
#SBATCH --job-name=medirai-multiclass
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
DATA=${DATA:-$SCRATCH/medirai/datasets_768}
SEED=${SEED:-0}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

TAG=multiclass_v2_seed${SEED}_$(date +%Y%m%d)
python classifier/train_multiclass.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run" \
  --seed "$SEED" --workers 8
python eval/baseline_report.py --out "eval/output/$TAG" --title "7-class diagnosis head, split v2 (seed $SEED)" \
  --setup "Narval A100, classifier/train_multiclass.py, batch 128, 30 epochs, class-weighted CE, best val AUROC of p(malignant)." \
  --runs mc="$TAG/run"
