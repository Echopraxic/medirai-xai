# MedirAI XAI Workspace

Code workspace for the Mitacs project *Explainable AI for Enhancing Deep Learning-Enabled Skin Cancer Detection* (Polytechnique Montréal × MedirAI, Sept 2026 – Jan 2027).

```
MedirAI/
├── README.md                 # this file
├── STUDENT2_PROJECT_PLAN.md  # Student 2 plan: TreeSHAP → concepts → LLM explanations
├── uncertaintyNet-main/      # ACTIVE: ResNet50 + variational/uncertainty training & testing (Student 1 base)
├── poc-nrc-main/             # ACTIVE: NRC ensemble (4 CNNs + BiomedCLIP/DermFoundation, fusion, Grad-CAM/SHAP/LIME)
│   ├── train_test_csvs/      #   all split / mixup / OOD CSVs (single copy)
│   └── archive/              #   superseded NRC code; not imported (see archive/README.md)
└── MedirAI Docs/             # reference documents (local only, not in git)
```

## What changed in the unification (2026-09-25)

The first commit (`Snapshot: original codebase as received`) holds everything exactly as received, so any removed file can be restored with `git checkout <snapshot> -- <path>`.

| Change | Why |
|---|---|
| Removed `poly-uncertainty-models-main/` | Byte-for-byte copy of `poc-nrc-main/` (164 MB) |
| Removed 23 files from `poc-nrc-main/ensemble_v1/` | Byte-identical to copies in `train_test_csvs/`, `development/`, `deprecated/`, `meta/`, or the root (~100 MB of duplicate CSVs) |
| Moved the rest of `ensemble_v1/`, `development/`, `deprecated/`, `ensemble_foundation_fusion.py`, and the `foundational_models/*.py` stubs into `poc-nrc-main/archive/` | Superseded by root-level modules; kept for reference |
| Moved `ensemble_v1/utils/hand_filtered_malig_egs.csv` → `train_test_csvs/` | Unique data-curation artifact, belongs with the other splits |
| Removed `data_loader.py`, a duplicate sample image, and `uncertaintyNet-main/sprint_updates_zabboud.pdf` | Identical to `data_downloader.py`, `sample_data/sample_img.jpg`, and the copy in `MedirAI Docs/` |
| `global_fusion.py` now reads `./train_test_csvs/test_EQ.csv` | Its old path pointed into the removed `ensemble_v1/` CSVs |

No model, training or evaluation logic was changed in this step.

The top-level folder names keep their upstream GitHub repo names (`poc-nrc`, `uncertaintyNet`) so they can still be diffed against upstream.

## Running

- Training and evaluation run on MedirAI's GCP VMs (GPU). The local machine is for editing only.
- uncertaintyNet: `python train.py --config medir.json` then `python test.py --run <run_dir>`.
- NRC: entry points are the `if __name__ == '__main__'` blocks in `medirai_inference.py` and `medirai_dnn_explainability_engine.py`.
- Both expect datasets outside the repo (`datasets/ISIC_clinical/metadata.csv` for uncertaintyNet; `../../../data/...` paths in the NRC CSVs).
