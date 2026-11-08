#!/bin/bash
#SBATCH --job-name=medirai-oof-fir
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# C1 out-of-fold targets on Fir (plan 2026-11-04; Narval and Rorqual are in maintenance until 2026-11-11 18:15).
# Same recipe as slurm/narval_stage_b.sh ARM=kdv8f_oof (final model kdv8f_mel, seed 3, split v8f without fold K).
#   for k in 0 1 2 3 4; do sbatch --account=def-omar12_gpu --export=ALL,K=$k slurm/fir_oof.sh; done
# Needs ~/medirai-xai (code, .venv, 12 teacher runs under uncertaintyNet-main/output), the torchvision ResNet50 weights in
# ~/.cache/torch/hub/checkpoints and the 12,252 images in $DATA/ISIC_clinical. Output: uncertaintyNet-main/output/<TAG>/run/.
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v4}
K=${K:?set K=0..4}
SEED=${SEED:-3}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
module load python/3.12.4
module load cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

T=narval_stageb
TEACHERS=rorqual_v2_lr3e-4_20261014/resnet50_det/run,narval_v2_seed1_lr3e-4_20261018/resnet50_det/run,${T}_ref_s2_20261021/resnet50_det/run
for a in hyg distill photo; do for s in 0 1 2; do TEACHERS=$TEACHERS,${T}_${a}_s${s}_20261021/run; done; done
TAG=narval_stageb_kdv8f_oof_k${K}_s${SEED}_$(date +%Y%m%d)   # same tag shape as the Narval jobs: eval/surrogate_c1b.py globs it

python classifier/train_sprint.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run" \
  --lr 3e-4 --epochs 30 --optimizer adamw --label_smoothing 0.1 --ema 0.999 --offline --workers 8 \
  --seed "$SEED" --teacher_runs "$TEACHERS" --kd_alpha 0.7 --kd_T 2 \
  --split_file "isic_v8f_oof_k${K}.csv" --mel_head 0.5
