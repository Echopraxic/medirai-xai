#!/bin/bash
#SBATCH --job-name=medirai-lesion-contrast
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Darkness-contrast arm LC (prereg draft LESION_CONTRAST_PREREG_DRAFT_2026-11-06.md): the kdv8f_mel recipe (F1, the
# adopted final model's arm) plus classifier/train_sprint.py --lesion_contrast 0.5. Same split, teachers and settings
# as narval_stageb_kdv8f_mel_s*_20261030, so the 5 F1 seeds are the control. No locked set is read.
#   STEP=masks   clinical UNet masks for the split's images (once, before any seed)
#   STEP=train   one seed: train, selection score, 8-view TTA (incl. the darker-skin dev set), darkness contrast on it
#
#   sbatch --account=def-omar12_gpu --export=ALL,STEP=masks slurm/narval_lesion_contrast.sh
#   for s in 0 1 2 3 4; do sbatch --account=def-omar12_gpu --export=ALL,STEP=train,SEED=$s slurm/narval_lesion_contrast.sh; done
# The teachers and the F1 runs live under the repo that trained them: set REPO to it if it isn't yours.
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
SRC=$SCRATCH/medirai/datasets_768
MIDAS=${MIDAS:-$SCRATCH/medirai/external/mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer/files_768}
DATA=${DATA:-$SCRATCH/medirai/datasets_768_v4}
MASKS=${MASKS:-$DATA/ISIC_clinical_masks/unet_clinical}
SPLIT=isic_v8f_v6_plus_f17k.csv
STEP=${STEP:?set STEP=masks|train}
SEED=${SEED:-0}
P=${P:-0.5}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export HF_HOME=${HF_HOME:-$SCRATCH/hf} HF_HUB_OFFLINE=1
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

case "$STEP" in
  masks)
    python segmentation/predict_unet.py --ckpt segmentation/output/unet_r18_clinical/best.pt \
      --images "$DATA/ISIC_clinical" --out "$MASKS" --split "$SPLIT" ;;
  train)
    TAG=narval_stageb_kdv8f_mel_lc_s${SEED}_$(date +%Y%m%d)
    T=narval_stageb
    TEACHERS=rorqual_v2_lr3e-4_20261014/resnet50_det/run,narval_v2_seed1_lr3e-4_20261018/resnet50_det/run,${T}_ref_s2_20261021/resnet50_det/run
    for a in hyg distill photo; do for s in 0 1 2; do TEACHERS=$TEACHERS,${T}_${a}_s${s}_20261021/run; done; done
    python classifier/train_sprint.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run" \
      --split_file "$SPLIT" --mel_head 0.5 --lr 3e-4 --epochs 30 --optimizer adamw --label_smoothing 0.1 --ema 0.999 \
      --offline --workers 8 --seed "$SEED" --teacher_runs "$TEACHERS" --kd_alpha 0.7 --kd_T 2 \
      --lesion_contrast "$P" --contrast_masks "$MASKS"
    python eval/selection_score.py --run "uncertaintyNet-main/output/$TAG/run" --isic_dir "$SRC/ISIC_clinical" \
      --midas_dir "$MIDAS"
    python eval/tta_scores.py --run "$TAG/run" --isic_dir "$SRC/ISIC_clinical" --midas_dir "$MIDAS" \
      --dsd_dir "$DATA/ISIC_clinical"
    python eval/darkness_contrast.py dsd --features features/output/external/dsd_features_v1.0.parquet \
      --tta "uncertaintyNet-main/output/$TAG/run/tta_scores.npz" \
      --out "eval/output/darkness_contrast/dsd_kdv8f_mel_lc_s${SEED}.json" ;;
  *) echo "unknown STEP $STEP"; exit 2 ;;
esac
