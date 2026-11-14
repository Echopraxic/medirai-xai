#!/bin/bash
#SBATCH --job-name=medirai-llm
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Prompt strategies on the final model's explanation inputs (feature table v1.1, MONET concepts; decision 2026-10-30),
# deterministic decoding, verified: template arm; Qwen2.5-7B x hybrid_v1 (reference), hybrid_fs_v1 (T5.1 b few-shot,
# + copy check) and hybrid_ng_v1 (T5.3 guardrail ablation); then the module end to end with the LLM (T8.1).
#   sbatch --account=def-omar12_gpu slurm/narval_llm.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
TREES=${TREES:-trees/output_v1.1_kdv8f_mel_s3}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PROMPTS=${PROMPTS:-"hybrid_v1 hybrid_fs_v1 hybrid_ng_v1"}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12 cuda arrow opencv
source "$REPO/.venv-llm/bin/activate"
cd "$REPO"

OUT=explain/output/narval_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu.txt"
pip freeze > "$OUT/pip_freeze.txt"

python explain/llm_generate.py --inputs "$TREES/explanation_inputs.jsonl" --backend template --out "$OUT"
for prompt in $PROMPTS; do
  echo "== $MODEL / $prompt"
  python explain/llm_generate.py --inputs "$TREES/explanation_inputs.jsonl" --backend vllm --model "$MODEL" \
    --prompt "$prompt" --out "$OUT"
done
python explain/fewshot_copy.py --gens "$OUT/$(basename "$MODEL")__hybrid_fs_v1/generations.jsonl" --prompt hybrid_fs_v1
python explain/run_module.py --trees "$TREES" --features features/output/feature_table_v1.1.parquet \
  --phrase_bank concepts/output/phrase_bank_v3.csv --model "$MODEL" --n 200 --out "$OUT/module_run"
