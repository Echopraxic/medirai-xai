# Pre-registration DRAFT: lesion-contrast augmentation arm (LC)

Shatha, 2026-11-06. **Draft for Michael**: the numbers marked *[proposed]* need agreeing before any LC run is looked at.
Selection data only; no locked set is read for this decision.

## Why

The darkness-contrast check (`eval/darkness_contrast.py`, darker-skin dev set, final model kdv8f_mel s3):
- Missed cancers are much darker than the surrounding skin (`C_darkness_vs_skin` median 13.4 missed vs 6.4 caught on
  FST V–VI; 13.2 vs 7.4 on FST IV).
- The ResNet's own score now falls as that feature rises within malignant lesions: Spearman −0.19 [−0.34, −0.03]
  on V–VI (kdv6_mel s1: −0.10, interval included 0).
- FST V–VI sensitivity at the v2-val 90%-sensitivity threshold is 0.74.

So the model has partly learned "darker than the skin = benign". LC tries to remove that cue from training without
new data.

## Arm

- **Control F1**: `narval_stageb_kdv8f_mel_s{0..4}_20261030` (the adopted recipe; 5 seeds exist).
- **LC**: the same recipe (split `isic_v8f_v6_plus_f17k.csv`, the 12-model KD teachers, `--mel_head 0.5`, AdamW, label
  smoothing 0.1, EMA 0.999, 30 epochs), plus `--lesion_contrast 0.5 --contrast_range 0.6,1.5`, seeds 0–4.
  With probability 0.5 the lesion (clinical UNet mask, feathered) is scaled by a log-uniform factor in [0.6, 1.5]
  against unchanged skin, **independently of the label**. Masks covering < 0.5% or > 60% of the image are skipped.
- Code: `classifier/train_sprint.py` (`LesionContrast`, `ContrastDataset`; tests in `tests/test_train_sprint.py`).
  Job: `slurm/narval_lesion_contrast.sh` (`STEP=masks` once, then `STEP=train SEED=s`).

## Data

v2 val, MIDAS dev_val (selection score S), and the darker-skin dev set. **Only FST V–VI** is used for tone endpoints:
the FST IV comparison images come from Fitzpatrick17k, which F1 also trains on (overlap check in
`slurm/narval_locked_kdv8f.sh`). If the check finds no overlap, FST IV is reported as well, as a secondary.

## Endpoints (5-seed means, LC vs F1; 95% intervals by bootstrap over seeds × lesion groups)

Primary:
1. **Shortcut**: Spearman(model score, `C_darkness_vs_skin`) within malignant FST V–VI. *[proposed]* LC moves it
   toward 0 by ≥ 0.05.
2. **FST V–VI sensitivity** at each run's own v2-val 90%-sensitivity threshold. *[proposed]* LC ≥ F1.

Non-inferiority (must all hold):
3. Selection score S: *[proposed]* LC ≥ F1 − 0.005.
4. DSD v2 AUROC: *[proposed]* lower bound of the LC − F1 difference > −0.02.
5. Val specificity at 90% sensitivity and melanoma recall: reported; *[proposed]* neither drops by more than 0.02.

Reported only: mask coverage of the train split (`contrast_mask_coverage` in each run's config), H2 (TreeSHAP of the
feature, missed vs caught), per-seed values.

## Decision rule

LC is adopted only if both primary endpoints and all non-inferiority checks pass. Then the single-seed rule (highest S)
picks the run, the explanation trees are refit on it, and it gets one final locked look (logged in `looks.csv`).
Otherwise kdv8f_mel s3 stays the final model and LC is reported as a negative result.

## Known risks

- Lesion darkness is also a real clinical cue (melanoma is often dark). Randomising it may cost accuracy; checks 3–5
  are there for that.
- The clinical UNet fails on wide phone shots; those images are skipped, so the augmentation mostly reaches
  close-ups. Coverage is reported.
- LC still trains on Fitzpatrick17k, so it is research-only like F1. If it works, the same augmentation goes into the
  commercial-safe retrain.

## Order

1. Wait for the kdv8f_mel s3 locked look and the FST IV overlap check.
2. Agree the *[proposed]* numbers with Michael and freeze this file.
3. Run `STEP=masks`, then the 5 LC seeds; run `eval/darkness_contrast.py dsd` on the 5 F1 seeds for the control values.
