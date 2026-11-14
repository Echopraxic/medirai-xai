#!/bin/bash
# D5 automated validation beyond the rule-based verifier (plan T6.1-T6.3), one job per generator set:
#   counterfactual sensitivity/stability + repeat determinism (explain/counterfactuals.py),
#   prompt-paraphrase robustness (hybrid_v1 vs hybrid_v1p on the same cases),
#   similar-input consistency and the NLI claim check (explain/consistency.py, explain/nli_check.py).
# One-time: the NLI model must be in $HF_HOME (login node):
#   source .venv-llm/bin/activate && hf download MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
#
#   sbatch --account=def-omar12_gpu slurm/fir_llm_d5.sh
#   INPUTS=... GENERATORS="..." sbatch --account=def-omar12_gpu slurm/fir_llm_d5.sh
# One H100 80 GB holds Qwen3-32B in bf16 (VLLM_TP=1 default). Narval D5 (2 A100, warm cache) took 16.5 min; Fir cold vLLM starts take ~10 min each: 90 min limit.
#
#SBATCH --job-name=medirai-llm-d5-fir
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
INPUTS=${INPUTS:-$REPO/trees/output_v1.1_kdv8f_mel_s3/explanation_inputs.jsonl}  # final model; the v1.0 deferral file is not on Fir
GENERATORS=${GENERATORS:-"Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-32B"}
PER_STRATUM=${PER_STRATUM:-75}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/cache} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SCRATCH/cache/vllm}   # persistent: a cold vLLM start (kernel compile) took 10.5 min on Fir 2026-11-05
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12.4 cuda arrow opencv
# Startup fix (2026-11-06): importing vllm from the Lustre-hosted venv took ~45 min per job. Unpack a single tarball
# (login node: cd $REPO && tar -cf $SCRATCH/venv-llm.tar .venv-llm) to node-local disk and use it from there.
T0=$(date +%s)
tar -xf "${VENV_TAR:-$SCRATCH/venv-llm.tar}" -C "$SLURM_TMPDIR"
export VIRTUAL_ENV="$SLURM_TMPDIR/.venv-llm" PATH="$SLURM_TMPDIR/.venv-llm/bin:$PATH"
echo "venv staged to $SLURM_TMPDIR in $(( $(date +%s) - T0 )) s"
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
