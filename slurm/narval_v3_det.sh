#!/bin/bash
# Deterministic ResNet50 on split v3 (v2 clinical + ISIC biopsied dermoscopic images in TRAIN only; val/test = v2),
# so the result is directly comparable with the v2 det baseline. Needs the dermoscopic cache first:
#   python tools/download_isic_resized.py --csv splits/isic_v3_clinical_plus_derm.csv --modality dermoscopic \
#       --out $SCRATCH/medirai/datasets_768/ISIC_dermoscopic          (login node)
#
#   sbatch --account=def-omar12_gpu --export=ALL,CONFIG=medir_v3.json slurm/narval_v3_det.sh
#   sbatch --account=def-omar12_gpu --export=ALL,SEED=1 slurm/narval_v3_det.sh            # second seed
#
#SBATCH --job-name=medirai-v3-det
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
SRC=$SCRATCH/medirai/datasets_768
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v3}     # ISIC_clinical/ = symlinks to clinical + dermoscopic caches
CONFIG=${CONFIG:-medir_v3.json}
LR=${LR:-3e-4}
EPOCHS=${EPOCHS:-20}
SEED=${SEED:-}                                     # empty = config seed (0)
EXTRA=${EXTRA:-}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"

mkdir -p "$DATA/ISIC_clinical"
for d in ISIC_clinical ISIC_dermoscopic; do
  find "$SRC/$d" -maxdepth 1 -name '*.jpg' -exec ln -sf {} "$DATA/ISIC_clinical/" \;
done

cd "$REPO/uncertaintyNet-main"
TAG=narval_v3_${CONFIG%.json}_lr${LR}${SEED:+_s$SEED}_$(date +%Y%m%d)
if [ -n "$SEED" ]; then EXTRA="$EXTRA --seed $SEED"; fi
python train.py --config "$CONFIG" --dataset_path "$DATA" --lr "$LR" --epochs "$EPOCHS" --no_date --no_compile \
  --save_path "output/$TAG/resnet50_det" $EXTRA
cd "$REPO"
python eval/export_predictions.py --run "$TAG/resnet50_det/run" --dataset-path "$DATA"
python eval/baseline_report.py --out "eval/output/$TAG" \
  --title "Deterministic ResNet50 on split v3 (v2 + ISIC dermoscopic train images; v2 val/test)" \
  --setup "Narval A100, $CONFIG, lr $LR, $EPOCHS epochs, extra args: ${EXTRA:-none}. Tag: $TAG." \
  --runs det="$TAG/resnet50_det/run"
