# GPU jobs on Rorqual and Narval (Calcul Québec / Digital Research Alliance)

Only public data is used here. Every job needs `--account=<your allocation>` (ours: `def-omar12`). Compute nodes have
no internet, so models, weights and pip packages are fetched on a login node first.

**Which cluster.** Rorqual (H100) is where the scripts were written, but our account's fair-share there fell to ~0.00004
and jobs waited for days. Narval (A100 40 GB, account `def-omar12_gpu`) has the allocation free, so GPU runs since
2026-10-14 go there: add `--gpus-per-node=a100:1` (or `a100:2` for 32B models / the VLM, with `VLLM_TP=2` or
`--tp 2`) and `--account=def-omar12_gpu` on the `sbatch` line; they override the scripts' `h100:1` header. Narval is
set up the same way (code, `.venv`, `.venv-llm`, `$SCRATCH/hf`). Rorqual is down 2026-10-27 05:00 to 2026-11-03 04:00.

## Layout on the cluster

| What | Where |
|---|---|
| Code | `~/medirai-xai` (a `git archive` tarball of the repo or a clone; the jobs only need the files) |
| Images (768-px cache, split v2) | `$SCRATCH/medirai/datasets_768/ISIC_clinical/<isic_id>.jpg` |
| Derm7pt (CC BY-NC-ND, research only, never redistributed) | `$SCRATCH/medirai/derm7pt/release_v0` |
| Hugging Face models | `$SCRATCH/hf` (`HF_HOME`) |
| ImageNet weights | `~/.cache/torch` (`TORCH_HOME`) |

Copying code without git (the cluster has no access to the private repo):

```bash
git archive --format=tar.gz -o code.tgz HEAD uncertaintyNet-main splits eval trees features segmentation tests requirements.txt slurm
scp code.tgz rorqual:~/ && ssh rorqual 'cd ~/medirai-xai && tar -xzf ~/code.tgz'
```

## One-time setup (login node)

Two separate virtual environments, so vLLM's torch pin cannot break training.

```bash
# training venv: install every package into the venv (the arrow module only helps pip resolve pyarrow);
# the jobs then run with just `module load python/3.12 cuda`
cd ~/medirai-xai
module load python/3.12 cuda arrow
python -m venv .venv && source .venv/bin/activate
pip install --no-index "numpy==2.4.2" pillow pandas scipy "scikit-learn<1.8" matplotlib seaborn tqdm tabulate \
    h5py lightning monai torch torchvision pytest
python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"   # pre-fetch into TORCH_HOME

# LLM / VLM venv (.venv-llm) + model downloads
bash slurm/rorqual_llm_setup.sh
```

## Jobs

| Script | What it does | Time |
|---|---|---|
| `rorqual_baseline_v2.sh` | ResNet50 deterministic + both variational objectives on split v2 (`configs/medir_v2.json`: lr 1e-4, 30 epochs, cosine), then prediction export, baseline report and the per-image uncertainty/deferral table | ≤ 6 h |
| `rorqual_llm.sh` | Every generator × prompt on the test split with greedy decoding, the template arm on the same inputs, the verifier, and a comparison table | ≤ 3 h |
| `rorqual_vlm_concepts.sh` | Qwen2.5-VL answers Derm7pt concept questions (clinical and dermoscopic photos), scored against the expert labels | ≤ 4 h |
| `rorqual_baseline.sh` | Original split-v1 baseline (kept for reference) | ≤ 4 h |

```bash
sbatch --account=def-omar12 slurm/rorqual_baseline_v2.sh
# Narval: same scripts, A100s
sbatch --account=def-omar12_gpu --gpus-per-node=a100:1 --export=ALL,LR=3e-4 slurm/rorqual_baseline_v2.sh
INPUTS=... GENERATORS=Qwen/Qwen3-32B PROMPTS=hybrid_v1 VLLM_TP=2 sbatch --account=def-omar12_gpu --gpus-per-node=a100:2     --time=04:00:00 --export=ALL slurm/rorqual_llm.sh
sbatch --account=def-omar12 --export=ALL,LR=3e-4 slurm/rorqual_baseline_v2.sh          # learning-rate sweep point
INPUTS=$HOME/medirai-xai/trees/output_v1.0/explanation_inputs.jsonl PROMPTS="hybrid_v1" \
    GENERATORS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-32B" sbatch --account=def-omar12 slurm/rorqual_llm.sh
sbatch --account=def-omar12 --export=ALL,MODALITIES=derm,OUT=concepts/output/<existing_run> \
    slurm/rorqual_vlm_concepts.sh                                                         # resume a run
```

Outputs land under the code folder (`uncertaintyNet-main/output/`, `eval/output/`, `explain/output/`,
`concepts/output/`); copy them back for analysis.

## Pitfalls we hit

- **No Triton** in the Alliance torch wheel, so `torch.compile` fails at the first batch: training jobs pass `--no_compile`.
- **Don't load the `opencv` module for training jobs.** monai imports cv2 when it is available, and the opencv module
  brings a libhdf5 that breaks the wheelhouse h5py (`undefined symbol: H5T_IEEE_F16BE_g`).
- **Modules can silently provide packages.** A venv built with `arrow`/`opencv` loaded found numpy and pandas in those
  modules and failed under `python/3.12 cuda` alone. Build the venv with exactly the modules the job loads.
- **Case-sensitive paths.** Derm7pt's `meta.csv` and its folders disagree on case (`FCl` vs `FCL`); fine on Windows,
  fatal on Linux. `concepts/vlm_concepts.py` matches paths case-insensitively.
- **Offline model lookup needs `refs/main`.** A Hugging Face cache copied between clusters (or filled with
  `--revision <sha>` only) has no `refs/` folder, and vLLM with `HF_HUB_OFFLINE=1` then fails with "Cannot find an
  appropriate cached snapshot". Run `hf download <model>` once on the login node, or write the sha into
  `$HF_HOME/hub/models--<org>--<name>/refs/main`. (`du` on the cache shows a few MB because the blobs are symlinks:
  use `du -shL .../snapshots`.)
- **Windows line endings** make `sbatch` refuse a script; keep scripts LF.
- Laptop runs (`--batch_size 32 --grad_accum 4 --amp`) and cluster runs (batch 128, no AMP) differ slightly, because
  BatchNorm sees 32 instead of 128 images per step.
