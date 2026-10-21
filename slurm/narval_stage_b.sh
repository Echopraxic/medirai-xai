#!/bin/bash
#SBATCH --job-name=medirai-stage-b
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Classifier sprint stage B (prereg 2026-10-21): one deterministic ResNet50 per job, then the pre-registered
# selection score (eval/selection_score.py: split v2 val + MIDAS dev_val, with TTA). lr 3e-4, 30 epochs (v2 recipe).
#   ARM=ref     split v2 (reference recipe; seeds 0/1 exist, run SEED=2)
#   ARM=v4      split v4 = v2 + MIDAS phone photos
#   ARM=v4bal   split v4 + diagnosis-balanced sampling (count^-0.5)
#   ARM=v4mc    split v4, 7-class diagnosis head (classifier/train_multiclass.py)
#   ARM=score RUN=<run folder under uncertaintyNet-main/output>   selection score of an existing run only
#
#   for a in v4 v4bal v4mc; do for s in 0 1 2; do
#     sbatch --account=def-omar12_gpu --export=ALL,ARM=$a,SEED=$s slurm/narval_stage_b.sh; done; done
#
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
SRC=$SCRATCH/medirai/datasets_768
MIDAS=${MIDAS:-$SCRATCH/medirai/external/mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer/files_768}
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v4}    # ISIC_clinical/ = symlinks to the ISIC clinical cache + MIDAS photos
ARM=${ARM:?set ARM=ref|v4|v4bal|v4mc|score}
SEED=${SEED:-0}
LR=${LR:-3e-4}
EPOCHS=${EPOCHS:-30}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"

if [ ! -f "$DATA/.linked" ]; then   # build the symlink folder once (batched; one ln per file is slow on Lustre)
  mkdir -p "$DATA/ISIC_clinical"
  for d in "$SRC/ISIC_clinical" "$MIDAS"; do
    find "$d" -maxdepth 1 -name '*.jpg' -print0 | xargs -0 ln -sf -t "$DATA/ISIC_clinical/"
  done
  touch "$DATA/.linked"
fi

score() {  # $1 = run folder under uncertaintyNet-main/output
  cd "$REPO"
  python eval/selection_score.py --run "uncertaintyNet-main/output/$1" --isic_dir "$SRC/ISIC_clinical" --midas_dir "$MIDAS"
}

TAG=narval_stageb_${ARM}_s${SEED}_$(date +%Y%m%d)
case "$ARM" in
  score)
    score "${RUN:?set RUN}" ;;
  ref|v4|v4bal)
    CONFIG=medir_v2.json; EXTRA=""
    [ "$ARM" != ref ] && CONFIG=medir_v4.json
    [ "$ARM" = v4bal ] && EXTRA="--balance_diagnosis 0.5"
    cd "$REPO/uncertaintyNet-main"
    # --no_compile: the Alliance torch wheel ships without Triton
    python train.py --config "$CONFIG" --dataset_path "$DATA" --lr "$LR" --epochs "$EPOCHS" --seed "$SEED" \
      --no_date --no_compile --save_path "output/$TAG/resnet50_det" $EXTRA
    cd "$REPO"
    python eval/export_predictions.py --run "$TAG/resnet50_det/run" --dataset-path "$DATA"
    score "$TAG/resnet50_det/run" ;;
  v4mc)
    cd "$REPO"
    python classifier/train_multiclass.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run" \
      --seed "$SEED" --lr "$LR" --epochs "$EPOCHS" --workers 8 --split_file isic_v4_clinical_plus_midas.csv
    score "$TAG/run" ;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
