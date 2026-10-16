#!/bin/bash
# D5 automated validation beyond the rule-based verifier (plan T6.1-T6.3), one job per generator set:
#   counterfactual sensitivity/stability + repeat determinism (explain/counterfactuals.py),
#   prompt-paraphrase robustness (hybrid_v1 vs hybrid_v1p on the same cases),
#   similar-input consistency and the NLI claim check (explain/consistency.py, explain/nli_check.py).
# One-time: the NLI model must be in $HF_HOME (login node):
#   source .venv-llm/bin/activate && hf download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
#
#   sbatch --account=def-omar12 slurm/rorqual_llm_d5.sh
#   INPUTS=... GENERATORS="..." sbatch --account=def-omar12 slurm/rorqual_llm_d5.sh
# Narval (A100 40 GB): Qwen3-32B needs two GPUs with tensor parallelism, which the 7B model also runs on:
#   VLLM_TP=2 sbatch --account=def-omar12_gpu --gpus-per-node=a100:2 --time=06:00:00 --export=ALL slurm/rorqual_llm_d5.sh
#
#SBATCH --job-name=medirai-llm-d5
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
INPUTS=${INPUTS:-$REPO/trees/output_v1.0_detv2/explanation_inputs_deferral.jsonl}
GENERATORS=${GENERATORS:-"Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-32B"}
PER_STRATUM=${PER_STRATUM:-75}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12 cuda arrow opencv
source "$REPO/.venv-llm/bin/activate"
cd "$REPO/explain"

OUT=output/d5_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
python counterfactuals.py make --inputs "$INPUTS" --out "$OUT/counterfactual_inputs.jsonl" --per_stratum "$PER_STRATUM"

FAILED=0
python llm_generate.py --inputs "$OUT/counterfactual_inputs.jsonl" --backend template --out "$OUT"
for model in $GENERATORS; do
  name=${model##*/}
  for prompt in hybrid_v1 hybrid_v1p; do
    echo "== $model / $prompt"
    python llm_generate.py --inputs "$OUT/counterfactual_inputs.jsonl" --backend vllm --model "$model" \
      --prompt "$prompt" --out "$OUT" || { echo "FAILED: $model / $prompt"; FAILED=1; continue; }
  done
  G="$OUT/${name}__hybrid_v1/generations.jsonl"
  [ -f "$G" ] || continue
  python counterfactuals.py score "$G" || FAILED=1
  python consistency.py similar "$G" || FAILED=1
  [ -f "$OUT/${name}__hybrid_v1p/generations.jsonl" ] && \
    python consistency.py paraphrase "$G" "$OUT/${name}__hybrid_v1p/generations.jsonl" || true
  python nli_check.py score "$G" || FAILED=1
done
python counterfactuals.py score "$OUT/template__template_v1/generations.jsonl"
python nli_check.py calibrate --inputs "$INPUTS" --limit 300 --out "$OUT/nli_calibration.json"
exit $FAILED
