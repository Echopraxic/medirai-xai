# NRC ensemble split CSVs: composition and known problems

Source is inferred from `image_id`: `ISIC_*` = ISIC 2024 permissive (3D total-body-photo tiles); `melanoma_*` = Kaggle "Melanoma Skin Cancer Dataset of 10000 Images". Counts are rows as of 2026-09-25, shown as target 0 / target 1.

| File | Rows | ISIC-2024 (0 / 1) | Kaggle (0 / 1) | Notes |
|---|---:|---|---|---|
| `train_data_EQ.csv` | 4,644 | 2,322 / 294 | 0 / 2,028 | **Label ≈ source.** Every benign image is ISIC; 87% of malignant images are Kaggle |
| `train_data_EQ_new_negs.csv` | 4,644 | 1,322 / 294 | 1,000 / 2,028 | Partly de-confounded (1,000 Kaggle benign hand-filtered in) |
| `train_data_TESTING_EQ.csv` | 4,644 | 2,322 / 294 | 0 / 2,028 | Same composition as `train_data_EQ.csv` |
| `train_data_TESTING.csv` | 217,477 | 217,183 / 294 | — | Full ISIC-2024 pool. Contains the 100 ISIC images of `test_EQ.csv`; don't train on it and then test on `test_EQ` |
| `mixup_train_data.csv` | 100,000 | 50,000 / 6,375 | 0 / 43,625 | Mixup pairs built from `train_data_EQ.csv`; inherits its confound |
| `mixup_train_data_new_negs.csv` | 100,000 | 28,560 / 6,346 | 21,440 / 43,654 | Mixup pairs from the new-negatives set |
| `test_EQ.csv` | 200 | 100 / 0 | 0 / 100 | **Label is perfectly predicted by source.** Don't use it for headline numbers (CODEBASE_TODO P0-6) |
| `test_EQ_new_neg.csv` | 200 | — | 100 / 100 | Single-source (Kaggle only); no source confound, but says nothing about ISIC images |
| `ood_training.csv` | 49,971 | 10 pseudo-label bins | — | OOD pretraining set. 29 `test_EQ` images were removed on 2026-09-25 (P0-8) |
| `biomed_50k.csv`, `biomed_50k_percentile_bin.csv` | 49,971 | 49,908 / 63 | — | BiomedCLIP scores for the OOD work. Same 29 test images removed (P0-8) |
| `hand_filtered_malig_egs.csv` | 2,028 | — | — | `img_id` list of hand-filtered Kaggle malignant examples |

## Rules going forward

- The reference benchmark for every model (ResNet50, NRC CNNs, tree models) is the **lesion/patient-grouped ISIC clinical close-up split** in [`splits/isic_clinical_v1.csv`](../../splits/isic_clinical_v1.csv).
- Whenever sources are mixed in training or evaluation, report accuracy **per source** alongside the pooled number.
- Before trusting a mixed-source model, check whether a classifier can predict the *source* from the images. If it can, the model can exploit the confound.
- Image paths in these CSVs are relative (`../../../data/...`). Fix them per machine; don't commit rewritten copies.
