# Running the baseline on Rorqual (Calcul Québec / Digital Research Alliance)

Only public ISIC data is used here. Proprietary MedirAI images stay on MedirAI's GCP infrastructure.

## One-time setup (login node, which has internet access)

```bash
git clone https://github.com/Echopraxic/medirai-xai.git ~/medirai-xai && cd ~/medirai-xai
module load python/3.12 cuda
python -m venv .venv && source .venv/bin/activate
pip install --no-index torch torchvision          # Alliance wheelhouse
pip install -r requirements.txt                    # falls back to PyPI for anything not in the wheelhouse
python -m pytest tests -q

# images: download, then build the 768-px cache the jobs read from
python splits/check_images.py --data-dir $SCRATCH/medirai/ISIC_clinical --download
python tools/cache_images.py clinical --src $SCRATCH/medirai/ISIC_clinical --out $SCRATCH/medirai/datasets_768

# compute nodes have no internet: pre-fetch the ImageNet weights into TORCH_HOME
python -c "from torchvision.models import resnet50, ResNet50_Weights as W; resnet50(weights=W.IMAGENET1K_V2)"
```

## Jobs

```bash
sbatch --account=<def-yourpi> slurm/rorqual_baseline.sh
```

This trains the deterministic ResNet50 at batch size 128 with no gradient accumulation and no AMP. It then trains both variational-last-layer objectives:
- `std` adds the MC-sample loss spread (merged P1-4 fix);
- `elbo` is the plain ELBO, run with `--no_likelihood_std`.

Finally it exports per-image predictions and writes `eval/output/rorqual_<date>/baseline_report.md`.

The laptop runs used `--batch_size 32 --grad_accum 4 --amp --no_compile`. That gives the same effective batch size, but BatchNorm sees 32 images per step instead of 128. Small differences between the two sets of numbers are therefore expected.
