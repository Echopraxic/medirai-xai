#!/bin/bash
# LLM explanation runs (plan WS4/WS5): every generator x prompt strategy on the test split, deterministic
# decoding, verified with explain/verifier.py, plus the no-LLM template arm on the same inputs.
# One-time setup: slurm/rorqual_llm_setup.sh (login node).
#
#   sbatch --account=def-omar12 slurm/rorqual_llm.sh
#   INPUTS=... GENERATORS="..." PROMPTS="..." sbatch --account=def-omar12 slurm/rorqual_llm.sh   # overrides
#
#SBATCH --job-name=medirai-llm
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
INPUTS=${INPUTS:-$REPO/trees/output_v0.2/explanation_inputs.jsonl}
GENERATORS=${GENERATORS:-"mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct"}
PROMPTS=${PROMPTS:-"zs_v1 cite_v1"}
EXTRA=${EXTRA:-}                                   # e.g. "--limit 20" for a smoke test
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12 cuda arrow opencv
source "$REPO/.venv-llm/bin/activate"
cd "$REPO"

OUT=explain/output/rorqual_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
pip freeze > "$OUT/pip_freeze.txt"

python explain/llm_generate.py --inputs "$INPUTS" --backend template --out "$OUT" $EXTRA
FAILED=0
for model in $GENERATORS; do
  for prompt in $PROMPTS; do
    echo "== $model / $prompt"
    python explain/llm_generate.py --inputs "$INPUTS" --backend vllm --model "$model" --prompt "$prompt" --out "$OUT" $EXTRA \
      || { echo "FAILED: $model / $prompt"; FAILED=1; }
  done
done
python explain/compare_runs.py "$OUT"
exit $FAILED
