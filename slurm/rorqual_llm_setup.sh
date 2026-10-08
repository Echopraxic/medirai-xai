#!/bin/bash
# One-time setup for the LLM explanation runs on Rorqual. Run on a LOGIN node (compute nodes have no internet).
#
#   bash slurm/rorqual_llm_setup.sh
#
# Creates a separate venv (.venv-llm) so vLLM's torch pin cannot break the training venv, and downloads the
# open-weight models to $HF_HOME. Only Apache-2.0 models are fetched by default; Llama-3.1-8B is gated
# (accept its licence on huggingface.co and `hf auth login` first, then set WITH_LLAMA=1).
set -euo pipefail

REPO=${REPO:-$HOME/medirai-xai}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
MODELS=(mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct)
[ "${WITH_LLAMA:-0}" = 1 ] && MODELS+=(meta-llama/Llama-3.1-8B-Instruct)

module load python/3.12 cuda arrow opencv
cd "$REPO"
if [ ! -d .venv-llm ]; then
  python -m venv .venv-llm
fi
source .venv-llm/bin/activate
pip install --upgrade pip
# Alliance wheelhouse first (built for the cluster), PyPI as fallback
pip install --no-index vllm 2>/dev/null || pip install vllm
pip install --no-index transformers pandas numpy huggingface_hub 2>/dev/null || pip install transformers pandas numpy huggingface_hub
python -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

mkdir -p "$HF_HOME"
for m in "${MODELS[@]}"; do
  echo "== downloading $m"
  hf download "$m" --exclude "*.pth" --exclude "original/*" --exclude "consolidated*"
done
du -sh "$HF_HOME"
echo "Setup done. Copy trees/output_v0.2/explanation_inputs.jsonl to $REPO/trees/output_v0.2/ and run:"
echo "  sbatch --account=def-omar12 slurm/rorqual_llm.sh"
