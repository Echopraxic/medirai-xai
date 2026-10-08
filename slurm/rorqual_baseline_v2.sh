#!/bin/bash
# Re-baseline ResNet50 (deterministic + both VLL objectives) on split v2 (MSKCC excluded) with the
# un-under-fitted recipe from configs/medir_v2.json: lr 1e-4, 30 epochs, cosine annealing.
# One job per learning rate; pick the run with the best val AUROC, then check Val Sigma moved (P1-5).
#
#   sbatch --account=<def-yourpi> slurm/rorqual_baseline_v2.sh                       # lr 1e-4
#   sbatch --account=<def-yourpi> --export=ALL,LR=3e-4 slurm/rorqual_baseline_v2.sh  # sweep point
#
#SBATCH --job-name=medirai-baseline-v2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
DATA=${DATA:-$SCRATCH/medirai/datasets_768}       # contains ISIC_clinical/<isic_id>.jpg
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-30}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch} # ImageNet weights pre-fetched on a login node
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO/uncertaintyNet-main"

TAG=rorqual_v2_lr${LR}_$(date +%Y%m%d)
COMMON=(--config medir_v2.json --dataset_path "$DATA" --lr "$LR" --epochs "$EPOCHS" --no_date)
python train.py "${COMMON[@]}" --save_path "output/$TAG/resnet50_det"
DET="$TAG/resnet50_det/run"

for variant in std elbo; do
  extra=""; [ "$variant" = elbo ] && extra="--no_likelihood_std"
  python train.py "${COMMON[@]}" --save_path "output/$TAG/resnet50_vll_$variant" \
    --init_network "$DET" $extra --network_type partial_bayesian
done

cd "$REPO"
for run in resnet50_det resnet50_vll_std resnet50_vll_elbo; do
  python eval/export_predictions.py --run "$TAG/$run/run" --dataset-path "$DATA"
done
python eval/baseline_report.py --out "eval/output/$TAG" \
  --runs det="$TAG/resnet50_det/run" vll_std="$TAG/resnet50_vll_std/run" vll_elbo="$TAG/resnet50_vll_elbo/run"
