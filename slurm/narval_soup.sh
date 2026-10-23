#!/bin/bash
#SBATCH --job-name=medirai-soup
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# Stage C test: probability ensembles and uniform weight soups of the seed groups (eval/soup_ensemble.py), selection
# data only. The groups are the 3-seed arms of stage B (ref = split v2 reference recipe).
#   sbatch --account=def-omar12_gpu slurm/narval_soup.sh
set -euo pipefail
REPO=${REPO:-$HOME/medirai-xai}
SRC=$SCRATCH/medirai/datasets_768
MIDAS=${MIDAS:-$SCRATCH/medirai/external/mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer/files_768}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export HF_HOME=${HF_HOME:-$SCRATCH/hf} HF_HUB_OFFLINE=1
module load python/3.12 cuda
source "$REPO/.venv/bin/activate"
cd "$REPO"
S=narval_stageb
G() { a=$1; shift; printf '%s=' "$a"; for s in "$@"; do printf '%s,' "$S"_"$a"_s"$s"_20261021/run; done | sed 's/,$//'; }
python eval/soup_ensemble.py --isic_dir "$SRC/ISIC_clinical" --midas_dir "$MIDAS" --all_ensemble \
  --out eval/output/stage_c/soup_ensemble.json \
  --group "ref=rorqual_v2_lr3e-4_20261014/resnet50_det/run,narval_v2_seed1_lr3e-4_20261018/resnet50_det/run,${S}_ref_s2_20261021/resnet50_det/run" \
  --group "$(G hyg 0 1 2)" --group "$(G hyg384 0 1 2)" --group "$(G distill 0 1 2)" --group "$(G photo 0 1 2)"
