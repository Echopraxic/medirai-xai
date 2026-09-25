# Archive (not maintained)

Superseded NRC code and data, kept for reference only. Nothing in the active codebase imports from here, and the files are not expected to run as-is: their relative imports point at modules that were deduplicated.

| Path | What it is | Superseded by |
|---|---|---|
| `ensemble_v1/` | Dec 2024 snapshot of the DNN ensemble (trainer, data loader, models, older feature fusion) | Root-level `trainer.py`, `isic_data_loader.py`, `medirai_*.py`, `dnn_feature_fusion.py` |
| `development/` | Iteration 0–1 experiments: vanilla CNN, segmentation training (`segmentation_2.py` trains the UNet), convex-hull mask cleanup, OOD pretraining, inference metrics. Imports `ensemble_v1.*` | Root-level code; `pytorch_unet.py` holds the UNet definition |
| `deprecated/` | Models without the hidden MLP head, plus the first feature-fusion attempts | `medirai_models_w_hidden.py`, `dnn_feature_fusion.py` |
| `ensemble_foundation_fusion.py` | First global-fusion version. It imports `feature_fusion`, which only existed in `ensemble_v1/` | `global_fusion.py` |
| `foundational_models_stubs/` | Early BiomedCLIP/DermFoundation scripts (`biomed_clip.py` has a syntax error) | Root-level `biomed_clip.py`, `derm_foundation.py` |
| `data_csvs/mixup_train_data_alt_draw.csv` | A second random draw of the 100k mixup pairs (differs from `train_test_csvs/mixup_train_data.csv`). It's unclear which draw trained the saved models | — |

Removed rather than archived because they were byte-identical to a kept copy: the whole `poly-uncertainty-models-main/` repo, `ensemble_v1`'s duplicate CSVs and scripts, `data_loader.py` (== `data_downloader.py`), and a duplicate sample image. All of them are recoverable from the first git commit (`Snapshot: original codebase as received`).
