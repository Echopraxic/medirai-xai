#!/bin/bash
# Re-baseline ResNet50 (deterministic + both VLL objectives) on Rorqual (Calcul Quebec / Alliance) at the
# original batch size (128, no accumulation). See slurm/README.md for one-time setup (env, data, weights).
#
#   sbatch --account=<def-yourpi> slurm/rorqual_baseline.sh
#
#SBATCH --job-name=medirai-baseline
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
DATA=${DATA:-$SCRATCH/medirai/datasets_768}       # contains ISIC_clinical/<isic_id>.jpg
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch} # ImageNet weights pre-fetched on a login node
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO/uncertaintyNet-main"

TAG=rorqual_$(date +%Y%m%d)
python train.py --config medir.json --dataset_path "$DATA" --save_path "output/$TAG/resnet50_det" --no_date
DET="$TAG/resnet50_det/run"

for variant in std elbo; do
  extra=""; [ "$variant" = elbo ] && extra="--no_likelihood_std"
  python train.py --config medir.json --dataset_path "$DATA" --save_path "output/$TAG/resnet50_vll_$variant" --no_date \
    --init_network "$DET" $extra --network_type partial_bayesian
done

cd "$REPO"
for run in resnet50_det resnet50_vll_std resnet50_vll_elbo; do
  python eval/export_predictions.py --run "$TAG/$run/run" --dataset-path "$DATA"
done
python eval/baseline_report.py --out "eval/output/$TAG" \
  --runs det="$TAG/resnet50_det/run" vll_std="$TAG/resnet50_vll_std/run" vll_elbo="$TAG/resnet50_vll_elbo/run"
