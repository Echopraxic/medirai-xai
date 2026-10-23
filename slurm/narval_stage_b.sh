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
#   batch 2 (classifier/train_sprint.py; hygiene = AdamW + label smoothing 0.1 + EMA 0.999):
#   ARM=hyg | hyg384 (+384 px) | distill (+DINOv2 feature distillation) | distill_v4 (+ split v4) | photo (+ phone /
#       darker-skin photometric augmentation) | dinov2 (DINOv2-L fine-tune; comparison only, never the product model)
#   stage C (decision 2026-10-22): ARM=kd | kd_v4 (split v4: the teacher also labels the MIDAS phone photos):
#       one ResNet50 distilled from the 12-model ensemble (ref, hyg, distill, photo arms; 3 seeds each)
#   batch 4 (on the kd_v4 student, decision 2026-10-22): ARM=kdv4_mel (+ melanoma head) | kdv6 (+ Derm7pt/Asan/SNU
#       melanoma data, split v6) | kdv6_mel (both)
#   sprint part 2 (prereg 2026-10-23; 5 seeds, kdv6_mel recipe): ARM=kdv7_mel (S2: + Mendeley light/dark pairs as
#       training data, split v7) | kdv6_mel_cons (S4: split v6 + tone-consistency loss on the pairs); both also write
#       tta_scores.npz (needs splits/f17k_dsd_v1.csv from slurm/narval_sprint2_data.sh)
#   ARM=score RUN=<run folder under uncertaintyNet-main/output>   selection score of an existing run only
#   ARM=tta RUN=<run folder>   8-view TTA scores incl. the darker-skin dev set (eval/tta_scores.py) of an existing run
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
export HF_HOME=${HF_HOME:-$SCRATCH/hf} HF_HUB_OFFLINE=1        # DINOv2-L pre-downloaded on a login node
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

tta() {  # $1 = run folder under uncertaintyNet-main/output
  cd "$REPO"
  python eval/tta_scores.py --run "$1" --isic_dir "$SRC/ISIC_clinical" --midas_dir "$MIDAS" --dsd_dir "$DATA/ISIC_clinical"
}

TAG=narval_stageb_${ARM}_s${SEED}_$(date +%Y%m%d)
case "$ARM" in
  score)
    score "${RUN:?set RUN}" ;;
  tta)
    tta "${RUN:?set RUN}" ;;
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
  kd|kd_v4|kdv4_mel|kdv6|kdv6_mel|kdv7_mel|kdv6_mel_cons)
    T=narval_stageb
    TEACHERS=rorqual_v2_lr3e-4_20261014/resnet50_det/run,narval_v2_seed1_lr3e-4_20261018/resnet50_det/run,${T}_ref_s2_20261021/resnet50_det/run
    for a in hyg distill photo; do for s in 0 1 2; do TEACHERS=$TEACHERS,${T}_${a}_s${s}_20261021/run; done; done
    case "$ARM" in
      kd)       A=() ;;
      kd_v4)    A=(--split_file isic_v4_clinical_plus_midas.csv) ;;
      kdv4_mel) A=(--split_file isic_v4_clinical_plus_midas.csv --mel_head 0.5) ;;
      kdv6)     A=(--split_file isic_v6_v4_plus_melanoma.csv) ;;
      kdv6_mel) A=(--split_file isic_v6_v4_plus_melanoma.csv --mel_head 0.5) ;;
      kdv7_mel) A=(--split_file isic_v7_v6_plus_mendeley.csv --mel_head 0.5) ;;
      kdv6_mel_cons) A=(--split_file isic_v6_v4_plus_melanoma.csv --mel_head 0.5 --consistency 1.0
                        --pairs_csv mendeley_pairs_v1.csv) ;;
    esac
    cd "$REPO"
    python classifier/train_sprint.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run"       --lr "$LR" --epochs "$EPOCHS" --optimizer adamw --label_smoothing 0.1 --ema 0.999 --offline --workers 8       --seed "$SEED" --teacher_runs "$TEACHERS" --kd_alpha 0.7 --kd_T 2 "${A[@]}"
    score "$TAG/run"
    case "$ARM" in kdv7_mel|kdv6_mel_cons) tta "$TAG/run" ;; esac ;;
  hyg|hyg384|distill|distill_v4|photo|dinov2)
    HYG=(--optimizer adamw --label_smoothing 0.1 --ema 0.999 --offline --workers 8 --seed "$SEED")
    case "$ARM" in
      hyg)        A=() ;;
      hyg384)     A=(--input_size 384) ;;
      distill)    A=(--distill 1.0) ;;
      distill_v4) A=(--distill 1.0 --split_file isic_v4_clinical_plus_midas.csv) ;;
      photo)      A=(--photo_aug) ;;
      dinov2)     A=(--backbone dinov2 --lr 2e-5 --batch_size 32 --epochs 15) ;;
    esac
    cd "$REPO"
    python classifier/train_sprint.py --dataset-path "$DATA" --out "uncertaintyNet-main/output/$TAG/run"       --lr "$LR" --epochs "$EPOCHS" "${HYG[@]}" "${A[@]}"
    score "$TAG/run" ;;
  *) echo "unknown ARM $ARM"; exit 2 ;;
esac
