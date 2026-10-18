#!/bin/bash
# LLM-as-judge (plan T6.4, prereg H3) on the D5 base explanations (template, Qwen2.5-7B, Qwen3-32B hybrid_v1) plus
# planted-error controls; judges come from families other than the generators (Gemma, Llama).
# Weights pre-downloaded to $HF_HOME on a login node (Gemma needs an HF token with the licence accepted):
#   hf download google/gemma-3-27b-it ; hf download casperhansen/llama-3.3-70b-instruct-awq
#
#   JUDGE=google/gemma-3-27b-it sbatch --account=def-omar12_gpu --export=ALL slurm/narval_llm_judge.sh
#   JUDGE=casperhansen/llama-3.3-70b-instruct-awq sbatch --account=def-omar12_gpu --export=ALL slurm/narval_llm_judge.sh
#
#SBATCH --job-name=medirai-llm-judge
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
JUDGE=${JUDGE:-google/gemma-3-27b-it}
D5=${D5:-output/d5_20261016_5852520}
OUTDIR=${OUTDIR:-output/judge_$(date +%Y%m%d)}
export HF_HOME=${HF_HOME:-$SCRATCH/hf} VLLM_TP=${VLLM_TP:-2}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
module load python/3.12 cuda arrow opencv
source "$REPO/.venv-llm/bin/activate"
cd "$REPO/explain"

GENS=("$D5/template__template_v1/generations.jsonl" "$D5/Qwen3-32B__hybrid_v1/generations.jsonl"
      "$D5/Qwen2.5-7B-Instruct__hybrid_v1/generations.jsonl")
python llm_judge.py run --gens "${GENS[@]}" --judge "$JUDGE" --out "$OUTDIR"
[ -f "$OUTDIR/human_ratings/rating_sheet.csv" ] || python llm_judge.py sheet --gens "${GENS[@]}" --out "$OUTDIR/human_ratings"
cp "${SLURM_SUBMIT_DIR:-.}/${SLURM_JOB_NAME:-judge}-${SLURM_JOB_ID:-local}.out" "$OUTDIR/${JUDGE##*/}/" 2>/dev/null || true
