#!/bin/bash
# FIR VERSION of narval_llm.sh (first Fir run timed out at 30 min: each cold vLLM start is ~10 min until the scratch kernel cache is warm; 60 min limit so the job backfills, no ReqNodeNotAvail before the 2026-11-17 07:00 outage).
#SBATCH --job-name=medirai-llm-fir
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Prompt strategies on the final model's explanation inputs (feature table v1.1, MONET concepts; decision 2026-10-30),
# deterministic decoding, verified: template arm; Qwen2.5-7B x hybrid_v1 (reference), hybrid_fs_v1 (T5.1 b few-shot,
# + copy check) and hybrid_ng_v1 (T5.3 guardrail ablation); then the module end to end with the LLM (T8.1).
#   sbatch --account=def-omar12_gpu slurm/fir_llm.sh
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
TREES=${TREES:-trees/output_v1.1_kdv8f_mel_s3}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
PROMPTS=${PROMPTS:-"hybrid_v1 hybrid_fs_v1 hybrid_ng_v1"}
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
cd "$REPO"

OUT=explain/output/fir_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}
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
