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
CONFIG=${CONFIG:-medir_v2.json}                    # medir_v2_384.json: 384-px input
NAME=${NAME:-rorqual_v2}
EXTRA=${EXTRA:-}                                   # e.g. "--batch_size 64 --grad_accum 2 --amp" at 384 px
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch} # ImageNet weights pre-fetched on a login node
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO/uncertaintyNet-main"

TAG=${NAME}_lr${LR}_$(date +%Y%m%d)
# --no_compile: the Alliance torch wheel ships without Triton, so torch.compile fails at the first batch
COMMON=(--config "$CONFIG" --dataset_path "$DATA" --lr "$LR" --epochs "$EPOCHS" --no_date --no_compile $EXTRA)
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
  --title "Baseline on split v2 (isic_clinical_v2, MSKCC excluded)" \
  --setup "Alliance cluster GPU runs ($CONFIG, 768-px image cache, extra args: ${EXTRA:-none}), 3 heads, $EPOCHS epochs, best-val checkpoint. Tag: $TAG." \
  --runs det="$TAG/resnet50_det/run" vll_std="$TAG/resnet50_vll_std/run" vll_elbo="$TAG/resnet50_vll_elbo/run"

# S1-4: per-image uncertainty + agreement/deferral table (adds S1's binary tree if its predictions exist)
U=uncertaintyNet-main/output/$TAG
RUNS=(det="$U/resnet50_det/run/predictions.csv" vll_std="$U/resnet50_vll_std/run/predictions.csv"
      vll_elbo="$U/resnet50_vll_elbo/run/predictions.csv")
TREE=trees/output/s1_binary_v0.2/tree_predictions.csv
[ -f "$TREE" ] && RUNS+=(tree="$TREE")
python eval/uncertainty_table.py --primary det --runs "${RUNS[@]}" --out "eval/output/$TAG/uncertainty"
