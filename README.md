# MedirAI XAI Workspace

Code for the Mitacs project *Explainable AI for Enhancing Deep Learning-Enabled Skin Cancer Detection*
(Polytechnique Montréal × MedirAI, Sept 2026 – Jan 2027). Two streams share one pipeline:

- **Student 1:** ResNet50 detection, uncertainty (variational layers, entropy), calibration, agreement-based deferral,
  and TreeSHAP on a binary tree to find what drives misclassification.
- **Student 2:** per-prediction top-3 TreeSHAP concepts from one-vs-all trees, mapped to dermatological concepts and
  narrated by an open-source LLM, then checked automatically and by clinicians. Plan: [STUDENT2_PROJECT_PLAN.md](STUDENT2_PROJECT_PLAN.md).

Only public data is used (ISIC clinical close-ups and public concept-validation sets). Only structured, de-identified
feature attributions ever reach a language model: no images, metadata or identifiers.

## Layout

```
splits/        persisted lesion/patient-grouped splits (v2 reference, v4 = v2 + MIDAS phone photos, MIDAS split) + builders
uncertaintyNet-main/  ResNet50 training/testing: deterministic and variational (train.py, test.py, configs/)
segmentation/  lesion masks: ResNet18-UNet, SAM second opinion, clinical fine-tune, gold set + annotation tool, QA flag
features/      concept-aligned handcrafted features (A/B/C, texture, shape) -> versioned feature table + dictionary
trees/         one-vs-all and binary XGBoost trees, surrogate of the ResNet50, TreeSHAP -> grouped top-3 concepts
explain/       number-free LLM input, prompts, template baseline, LLM generation (vLLM), rule-based verifier
concepts/      concept validation against ENHANCE, MILK10k MONET and Derm7pt (incl. a VLM concept checker)
eval/          prediction export, baseline + uncertainty/deferral tables, Grad-CAM arm, wrong-prediction study
tools/         image cache, ISIC and Redivis downloaders, results collector
classifier/    extra classifier heads: 7-class diagnosis head, foundation-model linear probes
slurm/         Rorqual / Narval (Calcul Québec) job scripts; see slurm/README.md
tests/         CPU regression tests (python -m pytest tests -q)
poc-nrc-main/  legacy NRC CNN/foundation-model ensemble (P0-fixed; models must be retrained before reuse)
```

## Versions everything is keyed to

| What | Current | Notes |
|---|---|---|
| Split | `splits/isic_clinical_v2.csv` | 6,743 images, 70/10/20 grouped by lesion and patient, seed 42. v1 minus the 1,240 MSKCC images (unbiopsied, contact-like photos that formed a benign shortcut). Never re-split; new data becomes a new version |
| Lesion masks | clinical fine-tuned ResNet18-UNet | Chosen on 97 hand-traced gold masks (`segmentation/gold/`); quality flag `quality_flag_v1` |
| Feature table | `features/output/feature_table_v1.0.parquet` | 42 features; `mask_quality` and the other QA columns are never model inputs |
| Explained model | deterministic ResNet50 | Split v2, lr 3e-4, seed 0, with 8-view test-time augmentation (interim, until the classifier sprint's model is frozen). Split v3 (v2 + ISIC dermoscopic images in train) was dropped: better internally, worse on external sets |
| Classifier evaluation | locked external sets + selection data | DDI, DDI-2, MCR-SL and ISIC clinical overview are scored only for final candidates (`eval/external_test.py --final`, every look logged). Model choices use split v2 val + MIDAS dev_val (`eval/selection_score.py`) |

Generated files (images, masks, feature tables, model outputs) are not in git; each script below rebuilds them.

## Pipeline

```bash
# 0. images: download the split's ISIC images, then build the 768-px cache every stage reads
python splits/check_images.py --data-dir uncertaintyNet-main/datasets/ISIC_clinical --download
python tools/cache_images.py clinical --src uncertaintyNet-main/datasets/ISIC_clinical --out uncertaintyNet-main/datasets_768

# 1. classifier (S1): train, export per-image predictions keyed by isic_id, report
#    (<run_dir> = run folder relative to uncertaintyNet-main/output, e.g. <tag>/resnet50_det/run)
cd uncertaintyNet-main && python train.py --config medir_v2.json && cd ..
python eval/export_predictions.py --run <run_dir> --dataset-path uncertaintyNet-main/datasets_768
python eval/baseline_report.py --runs det=<run_dir> --out eval/output/<tag>
python eval/uncertainty_table.py --primary det --runs det=<...>/predictions.csv --out eval/output/<tag>/uncertainty

# 2. masks: UNet on ISIC 2018, SAM, clinical fine-tune, quality flag, gold-set check
python segmentation/train_unet.py --data uncertaintyNet-main/datasets_768/ISIC2018 --out segmentation/output/unet_r18
python segmentation/finetune_clinical.py --init segmentation/output/unet_r18/best.pt \
    --isic2018 uncertaintyNet-main/datasets_768/ISIC2018 --out segmentation/output/unet_r18_clinical
python segmentation/predict_unet.py --ckpt segmentation/output/unet_r18_clinical/best.pt \
    --images uncertaintyNet-main/datasets_768/ISIC_clinical --out uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical
python segmentation/mask_quality.py --images uncertaintyNet-main/datasets_768/ISIC_clinical \
    --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks --primary unet_clinical --flag v1 \
    --split isic_clinical_v2.csv --out segmentation/output/qa_clinical
python segmentation/gold_eval.py --methods unet_clinical=uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical

# 3. features, trees, TreeSHAP top-3
python features/build_table.py --images uncertaintyNet-main/datasets_768/ISIC_clinical \
    --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical \
    --quality segmentation/output/qa_clinical/mask_quality.csv --out features/output/feature_table_v1.0.parquet
python trees/train_trees.py --features features/output/feature_table_v1.0.parquet --predictions <run_dir>/predictions.csv --out trees/output_v1.0
python trees/treeshap_top3.py --features features/output/feature_table_v1.0.parquet --trees trees/output_v1.0
python trees/s1_binary_tree.py                     # S1's ground-truth binary tree on the same table

# 4. explanations: template baseline locally; LLMs on a GPU (slurm/rorqual_llm.sh)
python explain/attach_deferral.py --inputs trees/output_v1.0/explanation_inputs.jsonl --table eval/output/<baseline_run>/uncertainty/uncertainty_table.csv \
    --out trees/output_v1.0/explanation_inputs_deferral.jsonl   # optional: S1-4 deferral flag as the "uncertain" signal
python explain/llm_generate.py --inputs trees/output_v1.0/explanation_inputs.jsonl --backend template --out explain/output
python explain/compare_runs.py explain/output/<run_folder>
# D5 checks (counterfactuals, consistency, NLI) and the LLM judges run on GPU: slurm/rorqual_llm_d5.sh, slurm/narval_llm_judge.sh

# 5. concept validation and the wrong-prediction / Grad-CAM studies
python concepts/validate_enhance.py
python concepts/validate_monet.py --features features/output/feature_table_v1.0.parquet
python eval/gradcam_arm.py --run <run_dir> --dataset-path uncertaintyNet-main/datasets_768 \
    --masks uncertaintyNet-main/datasets_768/ISIC_clinical_masks/unet_clinical --out eval/output/gradcam
python eval/wrong_prediction_study.py --inputs trees/output_v1.0/explanation_inputs.jsonl \
    --gradcam eval/output/gradcam/gradcam_metrics.csv --predictions <run_dir>/predictions.csv --out eval/output/c6

# 6. classifier sprint: external data, splits, selection score (GPU jobs: slurm/narval_stage_b.sh)
python tools/download_redivis.py --out <dir> <redivis dataset refs>      # MRA-MIDAS, DDI, DDI-2 (Narval login node)
python splits/make_midas_split.py && python splits/make_split_v4.py
python eval/selection_score.py --run <run_dir>                            # split v2 val + MIDAS dev_val, with TTA
python eval/external_test.py --source midas --runs <run_dir> ...          # locked sets need --final '<reason>'

# 7. collect every headline number into a local results page (RESULTS.md, not committed)
python tools/make_results.py
```

Each script's docstring lists its full options and outputs.

## Environment and compute

- Python 3.12 (3.13 works locally). `pip install -r requirements.txt` after installing torch/torchvision for your CUDA;
  `requirements-nrc.txt` adds the NRC ensemble's dependencies.
- GPU work runs on the Alliance clusters (Narval A100 by default, Rorqual H100) through `slurm/`; see
  [slurm/README.md](slurm/README.md). A 4 GB laptop GPU can train with `--batch_size 32 --grad_accum 4 --amp --no_compile`.
- Data licences: most ISIC images are CC-BY-NC and the Stanford AIMI sets are research-only. Licence columns are kept per
  image so a commercial-safe subset can be selected.
- Reproducibility: seeds are pinned; the split, feature table and prompts are versioned; every LLM run logs the model,
  prompt version and hash, decoding parameters and input hash.

## History

The first commit (`d2c62ee`, *Snapshot: original codebase as received*) holds everything as received. The second
removed a byte-identical copy of the NRC repo and duplicate CSVs and moved superseded NRC code to
`poc-nrc-main/archive/` (never imported). The P0 fixes (`ce71703`) replaced the per-image split with the grouped,
persisted split and removed several leakage paths and broken model heads, so numbers produced before them are not
comparable.
