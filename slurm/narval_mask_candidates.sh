#!/bin/bash
#SBATCH --job-name=medirai-mask-candidates
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
# Segmentation candidates on the hand-traced gold set only (plan SEGMENTATION_PLAN_2026-11-09.md, stage 1): SAM
# prompted with the clinical UNet's box, three UNet+SAM fusions, all scored by segmentation/gold_eval.py next to the
# current masks. Choose on the `tune` half; the `report` half is read for the chosen candidate only.
#   sbatch --account=def-omar12_gpu slurm/narval_mask_candidates.sh
#   SAM_MODEL=vit_h SAM_CKPT=<path to sam_vit_h_4b8939.pth> sbatch ...   (larger SAM, if the weights are on Narval)
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
IMAGES=${IMAGES:-$SCRATCH/medirai/datasets_768/ISIC_clinical}
M=uncertaintyNet-main/datasets_768/ISIC_clinical_masks       # relative to the repo, as gold_eval.py expects
SAM_MODEL=${SAM_MODEL:-vit_b}
SAM_CKPT=${SAM_CKPT:-uncertaintyNet-main/datasets/sam/sam_vit_b_01ec64.pth}
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"

G=$M/gold_candidates
rm -rf "$G" && mkdir -p "$G/unet_clinical"
tail -n +2 segmentation/gold/gold_set.csv | cut -d, -f1 | while read -r i; do
  [ -f "$M/unet_clinical/$i.png" ] && cp "$M/unet_clinical/$i.png" "$G/unet_clinical/"
done

python segmentation/sam_masks.py clinical --images "$IMAGES" --ckpt "$SAM_CKPT" --model "$SAM_MODEL" \
  --ids segmentation/gold/gold_set.csv --unet_masks "$G/unet_clinical" --out "$G/sam_box_$SAM_MODEL"
for rule in and or agree; do
  python segmentation/fuse_masks.py --a "$G/unet_clinical" --b "$G/sam_box_$SAM_MODEL" --rule "$rule" \
    --out "$G/fuse_${rule}_$SAM_MODEL"
done

python segmentation/gold_eval.py --methods unet_clinical=$M/unet_clinical sam=$M/sam \
  "sam_box_$SAM_MODEL=$G/sam_box_$SAM_MODEL" "fuse_and_$SAM_MODEL=$G/fuse_and_$SAM_MODEL" \
  "fuse_or_$SAM_MODEL=$G/fuse_or_$SAM_MODEL" "fuse_agree_$SAM_MODEL=$G/fuse_agree_$SAM_MODEL"
