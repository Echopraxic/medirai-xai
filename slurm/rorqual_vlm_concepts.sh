#!/bin/bash
# VLM concept-checker on Derm7pt (concepts/vlm_concepts.py): every model x {clinical, dermoscopic} photo.
# Derm7pt (CC BY-NC-ND, research use) must be copied to $DERM7PT first; it is never redistributed.
#
#   sbatch --account=def-omar12 slurm/rorqual_vlm_concepts.sh
#   MODELS="google/medgemma-4b-it google/medgemma-27b-it" sbatch ...   # after accepting MedGemma's terms + hf auth login
#
#SBATCH --job-name=medirai-vlm
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
DERM7PT=${DERM7PT:-$SCRATCH/medirai/derm7pt/release_v0}
MODELS=${MODELS:-"Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-32B-Instruct"}
MODALITIES=${MODALITIES:-"clinic derm"}
EXTRA=${EXTRA:-}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12 cuda arrow opencv
source "$REPO/.venv-llm/bin/activate"
cd "$REPO"

# OUT=<existing run folder> resumes it: cases already in answers.jsonl are skipped
OUT=${OUT:-concepts/output/vlm_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
FAILED=0
for model in $MODELS; do
  for modality in $MODALITIES; do
    echo "== $model / $modality"
    python concepts/vlm_concepts.py --model "$model" --modality "$modality" --derm7pt "$DERM7PT" --out "$OUT" $EXTRA \
      || { echo "FAILED: $model / $modality"; FAILED=1; }
  done
done
python concepts/eval_vlm_concepts.py "$OUT" --derm7pt "$DERM7PT" || true
exit $FAILED
