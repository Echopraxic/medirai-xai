# Segmentation plan: better lesion masks for the explainer

Shatha, 2026-11-09. **Draft for Michael.** On 11-06 he decided to keep the clinical UNet masks for the freeze
(note section 7, item 3), so stages 2–3 below reopen that decision and need his agreement. Stage 1 reads only the
hand-traced gold set and changes nothing downstream.

## Why masks are the lever

- 42% of split-v2 images get a `fail` mask flag (val 42.2%, test 40.8%). Clinical UNet on the gold set: Dice 0.713
  (reweighted), 37% of masks below Dice 0.8.
- Every handcrafted feature (A, B, C, T, S groups) is measured inside the mask; only the three MONET concepts are not.
- Explainer agreement with the ResNet on benign calls (kdv8f_mel s3, features v1.1, val): **85.8% with an ok mask,
  56.8% with a failed one** (n 127 / 37). The C1 benign gap (80.7% on test) comes mostly from failed masks.
- Explanation-side fixes don't close it (val, `trees/mask_aware_surrogate.py`): giving the tree the mask flag lifts
  failed-mask benign agreement from 0.514 to 0.568 (about 2 of 37 images); the UNet quality statistics add nothing.
  Basing failed-mask reasons on the MONET concepts alone leaves it at 56.8%. A wrong outline gives wrong
  measurements, and no rule applied afterwards recovers them.

## Weak points in the current pipeline

- The UNet predicts at 256 px (training size) on 768-px images, so fine borders are lost.
- SAM (ViT-B) is prompted with one point at the image centre, so off-centre lesions fail. Its masks only serve as the
  agreement signal in the flag.
- 97 gold masks (50 tune / 50 report): enough to choose between methods, too few to train on.

## Stage 1: no training, gold set only (1 short GPU job)

`slurm/narval_mask_candidates.sh`:
- **sam_box:** SAM prompted with the clinical UNet's box (grown 10%) plus a point at its centre
  (`segmentation/sam_masks.py --unet_masks`), falling back to the centre point when the UNet mask is empty or
  covers more than 85% of the frame. ViT-H if the weights are on Narval (`SAM_MODEL=vit_h`).
- **Fusions** of the UNet and sam_box (`segmentation/fuse_masks.py`): `and`, `or`, `agree` (overlap if Dice ≥ 0.6,
  else UNet).
- Scored by `segmentation/gold_eval.py` next to unet_clinical and the centre-point SAM.

**Decision (tune half only)** *[proposed]*: a candidate goes forward if, against unet_clinical on the tune half, mean
Dice rises by ≥ 0.03 (paired bootstrap over images, interval excluding 0) and the share below Dice 0.8 falls. Pick the
best one; read the report half once, for that candidate only, and quote those numbers.

## Stage 2: only if stage 1 gains less than 0.03

- **Higher-resolution UNet:** fine-tune the clinical UNet at 512 px (`finetune_clinical.py --size 512`), same
  pseudo-label recipe.
- **More gold masks:** trace 100 more images with `segmentation/annotate.py`, sampled like the first 100 (one-third
  per flag, MILK/UFES mix). That doubles the tune half and allows a small fine-tune on true masks. About 2–3 hours of
  tracing (the first 97 took about 2 minutes each).

## Stage 3: if a candidate is adopted (needs Michael)

1. Masks for all images (split v2, the extra +E rows, external sets). Refit the `quality_flag_v1` thresholds on the
   tune half for the new masks.
2. Feature table v1.2 (same extractor, new masks), with v1.1's MONET columns carried over unchanged.
3. Refit the surrogate and TreeSHAP for kdv8f_mel s3; citation rule R1 unchanged.
4. **Success check, on val only** *[proposed]*: per-class agreement ≥ 85% on both classes; validated-reason share
   not lower than v1.1's 65.4%. Then one test look, as for 5c.
5. Knock-on effects to re-run: feature-validity wording (S1-9), the C2 concept checks on mask-based concepts,
   Grad-CAM mask overlap, and the masks used by the lesion-contrast training arm.

## Risks

- 50 tune images: wide intervals, and choosing among five candidates on them overfits a little. Hence the
  0.03 margin and the single report look.
- Better masks on the gold mix may not mean better masks on wide phone shots (the gold set is close-ups). Report
  MILK and UFES separately.
- Stage 3 touches most of the pipeline during the freeze. Doing it after the clinician sessions (Nov 30 – Dec 13)
  is the safer timing.
