# Student 2 Project Plan: Feature Selection for LLM-Based Explanation Generation

**Project:** Explainable AI for Enhancing Deep Learning-Enabled Skin Cancer Detection (Mitacs Accelerate, IT 38322)
**Student 2:** Michael McKenna · **Student 1:** Shatha Albeetar · **Supervisor:** Prof. Adel Abusitta (Polytechnique Montréal) · **Partner:** MedirAI (J.-F. Djoufak)
**Duration:** 4 months, **Mon 2026-09-21 → Wed 2027-01-20**.
- Month 1 = W1–4 (Sep 21 – Oct 18)
- Month 2 = W5–8 (Oct 19 – Nov 15)
- Month 3 = W9–12 (Nov 16 – Dec 13)
- Month 4 = W13–17 (Dec 14 – Jan 20). It includes the Polytechnique holiday closure (~Dec 23 – Jan 5), so plan roughly three working weeks.
**Last updated:** 2026-10-21

Legend for owners: **S2** = Student 2 (me) · **S1** = Student 1 · **M** = MedirAI · **A** = Academic supervisor / admin. Dependency IDs (S1-x, M-x, A-x) are defined in §6.

---

## 1. Goal and success criteria

**Goal:** For each prediction of MedirAI's ResNet50 skin-lesion classifier, produce a short, clinician-facing, natural-language explanation. The explanation must be:
- grounded in the top-3 discriminating features found by TreeSHAP,
- phrased in recognized dermatological concepts (ABCD rule, surface and texture terms),
- honest about uncertainty and deferral, and
- shown to be faithful and useful by automated checks and a small dermatologist review.

**Proposed success criteria** (to agree with Prof. Abusitta and MedirAI in week 2):

| # | Criterion | Proposed target |
|---|---|---|
| C1 | Surrogate fidelity: tree-model prediction agrees with ResNet50 on the test set | ≥ 85% agreement, reported per class |
| C2 | Feature→concept validity: concept scores agree with expert/public annotations (ENHANCE / PH2 / Derm7pt) | Spearman ρ ≥ 0.5 or Cohen's κ ≥ 0.4 per concept; concepts below this are dropped or flagged |
| C3 | Grounding: explanation claims that trace to a supplied feature | ≥ 98%, with 0 invented concepts in the audited sample |
| C4 | Direction correctness: "supports/argues against" matches the SHAP sign | ≥ 95% |
| C5 | Uncertainty honesty: deferred or uncertain cases are explicitly flagged in the text | 100% (rule-checked) |
| C6 | No false confidence: confident wording on *incorrect* predictions is no higher than on correct ones | Measured and reported |
| C7 | Clinician review: the natural-language (NL) explanation is rated ≥ Grad-CAM and ≥ label-only on clarity and trust | Descriptive result with inter-rater agreement (small N) |

---

## 2. Pipeline overview

```
            ┌──────────────── S1: ResNet50 (+ uncertainty / deferral) ─────────────┐
 image ─┬──►│ prediction, p(malignant), entropy measures, agreement-deferral flag │──┐
        │   └──────────────────────────────────────────────────────────────────────┘  │
        │   ┌──── SHARED (S1+S2): feature layer ───┐                                  │
        ├──►│ segmentation mask → handcrafted A/B/C/│                                  │
        │   │ D-pixel/texture/shape features (+ opt.│                                  │
        │   │ ResNet50 embedding PCs)  → feature    │                                  │
        │   │ table (parquet, keyed by isic_id)     │                                  │
        │   └───────────────┬───────────────────────┘                                  │
        │                   ▼                                                          │
        │   S1: binary tree model + TreeSHAP    S2: one-vs-all tree models + TreeSHAP  │
        │                                        │                                     │
        │                                        ▼                                     │
        │               S2: concept grouping → top-3 concepts (value, direction, size) │
        │                                        │                                     │
        │                                        ▼                                     │
        │               S2: concept map → clinical phrases (validated thresholds)      │
        │                                        │                                     │
        │                                        ▼                                     │
        │   S2: structured JSON (no image, no identifiers) ◄────────────────────────────┘
        │                                        │
        │                                        ▼
        │               S2: open-source LLM (Llama 3.x 8B / Mistral 7B, local on GCP)
        │                     → explanation + per-claim feature_id (JSON)
        │                                        │
        │                                        ▼
        │               S2: verifier (grounding, direction, uncertainty rules)
        │                                        │
        └──── Grad-CAM (S1) ─────►  S2: evaluation (auto + LLM-judge + clinician review)
```

**Design principles carried over from prior MedirAI work**:
- Convert numbers into clinical categories *before* the LLM. Llama handled raw feature values poorly in the prior pipeline.
- Thresholds must be validated, not guessed. The prior handcrafted color and texture rules scored only ~27–30%.
- Do all evaluation with deterministic decoding.
- Explicitly test on wrong predictions. Prior VLM-reads-heatmap explanations almost always "supported" the model.

---

## 3. Student 2 deliverables (from grant §2.2)

| ID | Deliverable | Due | Grant link |
|---|---|---|---|
| D1 | Shared feature-extraction module + versioned feature table + feature dictionary (co-owned with S1) | End M2 | "working feature attribution pipeline" |
| D2 | One-vs-all tree models + TreeSHAP ranking + top-3 concept selector | End M2 | "ranked, per-prediction feature sets" |
| D3 | Feature→dermatological-concept map with validation report | End M2 | "mapped to clinical concepts where feasible" |
| D4 | Prompt library + LLM explanation module (open-source LLM, local) | End M3 | "prototype module integrating the open-source LLM" |
| D5 | Automated validation protocol + results (faithfulness, consistency, LLM-as-judge) | M3–M4 | "LLM-as-judge… accuracy, coherence, consistency" |
| D6 | Clinician review study (protocol, materials, results vs Grad-CAM and label-only) | End M4 | "small-scale clinician review" |
| D7 | Integrated prototype (with S1's refined ResNet50), documentation, code handoff to MedirAI | M3 / M4 | Shared milestones |
| D8 | Contribution to the data-use / privacy / ethics summary | M4 | "summary of the data-use and privacy safeguards" |

---

## 4. Timeline at a glance

Aligned with the grant Gantt (§2.3). ★ marks a shared milestone.

| Week | Student 2 focus | Needs from others | Gives to others |
|---|---|---|---|
| **M1 · W1–2** (Sep 21 – Oct 4) | ☑ Docs/code review, plan, codebase unification + `CODEBASE_TODO.md` (local). Access; scope and success criteria; ethics query started; **P0 fixes with S1: grouped + persisted split, NRC import fix** | M-1, M-2, M-4, M-5, M-8, A-1, S1-1 | Split fix (P0-1/2) co-developed with S1 |
| **M1 · W3–4** (Oct 5 – 18) | S1 re-baseline on the new split (P0 done). Shared feature extraction v0; one-vs-all class definition; concept dictionary draft; LLM shortlist + serving set up | M-3, S1-2 | Feature schema → S1 |
| **M2 · W5–6** (Oct 19 – Nov 1) | Feature table v1; one-vs-all models + TreeSHAP; concept grouping + top-3 selector; input JSON schema | S1-3 (preliminary) | Feature table v1 → S1 |
| **M2 · W7–8** (Nov 2 – 15) | Concept validation vs ENHANCE/PH2/Derm7pt; surrogate fidelity; prompt strategies v1; clinician study protocol + form drafted | S1-3, S1-4, S1-5, M-6, M-7 | ★ **End M2 (Nov 15):** baselines + attribution outputs available to both streams |
| **M3 · W9–10** (Nov 16 – 29) | Prompt strategy comparison (deterministic); verifier; LLM-as-judge calibrated; case set for clinician review frozen | S1-6, S1-7, S1-9 (A-1 cleared 2026-10-09) | Explanations for deferred cases → S1 |
| **M3 · W11–12** (Nov 30 – Dec 13) | Integrated prototype; clinician review sessions run; automated evaluation on full test set | S1-8, M-7, M-10 | ★ **End M3 (Dec 13):** integrated prototype functional |
| **M4 · W13–14** (Dec 14 – 22) | Analysis: auto metrics + clinician results; baseline comparisons (label-only, Grad-CAM, template); error analysis on wrong/deferred cases | S1-8 (final model) | Results → joint report |
| *Holiday closure* (~Dec 23 – Jan 5) | Buffer only; plan no deliverables | — | — |
| **M4 · W16–17** (Jan 6 – 20) | Finalize module, docs, handoff; joint report; final Mitacs report | M-10 | ★ **End M4 (Jan 20):** clinician review done, joint report, code/docs delivered |

> **Note on the grant Gantt:** the Student 2 table repeats "Design top-feature selection method and map features to dermatological concepts" in Month 4. This looks like a copy error. This plan treats Month 4 as analysis, integration, and documentation. Confirm with Prof. Abusitta (see A-2).

---

## 5. Work breakdown

Status: ☐ not started · ◐ in progress · ☑ done

### WS0: Onboarding and setup (M1, W1–2)

- ◐ **T0.1** Request access to the prior-intern repos and model bucket (→ M-1). *Requested 2026-09-25; awaiting access.* Read `app/extract_features.py`, `app/prompts.json`, `app/skin_lesion_analyzer.py`, and the fine-tuning scripts.
- ◐ **T0.2** Get GCP VM + GPU access working over SSH/VS Code (→ M-2). *Requested 2026-09-25; awaiting access.* Create a Python 3.12 env with `xgboost`/`lightgbm`, `shap`, `scikit-image`, `opencv`, `mahotas`/`pyradiomics` (optional), and `vllm` or `transformers`. *2026-10-02: interim local GPU (RTX 3050, 4 GB) env working; Rorqual (Calcul Québec) job scripts in `slurm/`; GCP VM still pending.*
- ◐ **T0.3** Set up a shared GitHub repo/folder layout with S1 for the feature layer (`features/`, `trees/`, `explain/`, `eval/`), plus experiment logging (CSV or W&B; W&B was already used for LLaVA fine-tuning). *2026-10-02: `features/`, `trees/`, `explain/`, `eval/`, `segmentation/` created on branch `s2-baseline-masks-gradcam`; layout to confirm with S1.*
- ☐ **T0.4** Agree on the success criteria (§1) and the check-in cadence with the supervisor, S1, and MedirAI (→ M-9).
- ☑ **T0.5** Kick off the ethics question with the supervisor (→ A-1). *Answered 2026-10-09: Prof. Abusitta confirmed no Polytechnique ethics certification is required (secondary public data + clinician review of explanations). Record this in the D8 data-use summary.*
- ☑ **T0.6** Split file built (done by S2 on S1's behalf, 2026-09-25): [splits/isic_clinical_v1.csv](splits/isic_clinical_v1.csv) plus its report (`splits/isic_clinical_v1_report.md`, written by the split script).
  - Metadata for all 9,316 ISIC clinical close-ups was pulled from the public ISIC API into `splits/isic_clinical_closeup_metadata.csv`.
  - Diagnosis levels: `diagnosis_1` (Benign 3,196 / Malignant 4,787 / Indeterminate 854 / missing 479); `diagnosis_3` for one-vs-all. **BCC dominates the malignant class** (3,396); melanoma subtypes total ~500.
  - Useful for WS1/WS3: `clin_size_long_diam_mm` (**real mm diameter** for 1,229 images, mostly UFES), `fitzpatrick_skin_type` (2,457 images, 384 in types V–VI), anatomic site, age and sex.
- ☑ **T0.7** Codebase unification (2026-09-25): removed the duplicate repo and duplicate CSVs, archived superseded code, put the workspace under git, wrote `CODEBASE_TODO.md` (local).
- ◐ **T0.8** Stabilize the foundation with S1 before building on it (W2–W3). **Code-side P0 fixes done on 2026-09-25** (all except P0-6, which is partial, plus the new P0-17); still needs the GPU re-baseline by S1. Items from `CODEBASE_TODO.md` (local): *2026-10-02: S1's P1 fixes merged (PR #1) and covered by regression tests; provisional local re-baseline done (deterministic 0.828 / variational 0.837–0.840 test accuracy, AUROC 0.88–0.90); official numbers = S1's cluster re-run.*
  - **P0-1/2/3:** grouped + persisted split. This *is* S1-1, co-owned.
  - **P0-9:** NRC import fix. It blocks Grad-CAM.
  - **P0-10/11:** `test.py` and checkpoint loading.
  - Then S1 re-baselines ResNet50 on the new split, and that one number is what both streams cite.
  - ☑ Handoff Steps 1–4 prepared and verified locally (2026-09-25): the handoff note to S1 (local, archived), `splits/check_images.py`, `requirements*.txt`, and a git bundle. S1 still runs the tests, split check and image check on the VM.
  - S2-relevant P1 items: P1-15 (uniform logits export), P1-16 (Grad-CAM on the predicted class), P1-17 and P1-10 (mask quality and Dice).

### WS1: Shared feature layer (co-owned with S1) (M1 W3 → M2 W6)

- ☐ **T1.1** Agree on the feature schema with S1 (→ S1-2). One row per `isic_id` with columns `split`, labels, `mask_quality`, then the feature columns. Store as versioned parquet.
- ◐ **T1.2** Masks: get segmentation masks for the clinical close-ups (→ M-3). Apply the convex-hull cleanup from the NRC work (largest contour → hull). Compute a **mask-quality flag** (area ratio, border touching, confidence/uncertainty if Aloys's UNet provides it). Prior report: segmentation models "don't work with real (uncropped) images". *2026-10-02: v0 masks for all 7,983 images: ResNet18-UNet trained on ISIC 2018 (test Dice 0.894) + zero-shot SAM second opinion → ok/review/fail flag. Only ~51% ok/review: the dermoscopy-trained UNet over-segments small or faint clinical lesions. No convex hull (it erases border irregularity). Next: adapt to clinical photos with UNet–SAM agreement pseudo-labels; M-3 weights still wanted. 2026-10-07: gold set of 100 split-v2 test images (55 MILK / 45 UFES, stratified by flag and label, tune/report halves) + annotation tool (`segmentation/gold_select.py`, `segmentation/annotate.py`). 2026-10-09: gold set annotated by S1 (blind to model masks): 97 traced, 3 "no clear lesion" (all in the tune half), 16 "border uncertain". Report half, Dice reweighted to the test-set flag mix (fail 47% / ok 30% / review 22%): UNet 0.61 (58% of masks below Dice 0.8), SAM 0.64 (44%), clinical fine-tuned UNet 0.71 (37%). The flag works: UNet bad-mask rate 12% (ok) / 65% (review) / 91% (fail); UNet–SAM agreement predicts a bad mask at AUROC 0.88. All methods fail on border-uncertain lesions (Dice ≈ 0.39). Output: `segmentation/output/gold_eval/`. v1 masks = clinical fine-tuned UNet with a gold-calibrated flag (`quality_flag_v1`: UNet mean confidence in mask, SAM agreement, area, frame touch; fit on tune, checked on report: bad-mask rate ok 19% / review 64% / fail 56%). On split v2: ok 41% / review 17% / fail 42%, and 83% of fail masks are malignant lesions, so the flag must never be a model input.*
- ◐ **T1.3** Implement **concept-aligned** feature families. Start from `extract_features.py` and fix it where needed: *2026-10-02: v0.1 = 44 features (A/B/C/D, texture, shape) at a common 512-px working size. 2026-10-07: v0.2 = 42 features: Shades-of-Gray color constancy, lesion-scale resampling (Feret = 192 px), frame-relative size moved out of the model (`frame_*`, QA only). Correction: the MSKCC shortcut was color/texture, not D. Within one diagnosis, features still separate UFES from MILK (AUROC 0.86–0.91, mostly color: population and photography); reported per source.*
  - **A (asymmetry):** mask flip overlap about both principal axes, not just horizontal; color asymmetry across halves.
  - **B (border):** circularity, fractal dimension, convexity defects, border gradient sharpness (abrupt vs. fading edge).
  - **C (color):** count of clinically named colors (light/dark brown, black, blue-gray, red, white) in a perceptual space (CIELAB) instead of CSS3 nearest-name; color variance/entropy.
  - **D:** pixel Feret diameter, with the no-mm-calibration caveat.
  - **Texture/surface:** GLCM Haralick and LBP.
  - **Shape:** Hu moments, eccentricity, solidity.
  - Optional: PCA components of ResNet50 penultimate embeddings (→ S1-6).
- ☑ **T1.4** Unit tests on synthetic shapes and masks (a known-symmetric disc, a known number of colors) so every feature behaves as named. *2026-10-02: `tests/test_features.py` (symmetric disc, rotated ellipse, star border, two-color lesion, rough surface, resolution invariance).*
- ◐ **T1.5** Run on the full dataset on GCP → **feature table v1** plus a **feature dictionary** (name, concept group, units, direction meaning, valid range, extraction version). *(D1)* *2026-10-02: feature table v0.1 + dictionary + manifest built locally (329 failed-mask rows kept with an error flag); v1 after S1 review. 2026-10-07: v0.2 on split v2 (6,743 rows, 85 errors). 2026-10-09: S1 reviewed v0.2 before the v1 freeze. The 85 errors are all "mask too small": the original UNet returned an empty mask (all flagged fail), 63 of them malignant (54 BCC), 16 in test; the clinical fine-tuned UNet gives a non-empty mask for all 85. `mask_quality` (+ UNet–SAM agreement stats) is already in the parquet, but not in the dictionary CSV. No D feature by design (frame-relative size was a framing shortcut; mm only for UFES). Decided 2026-10-09 (§8): v1 uses the clinical fine-tuned UNet masks (no rows dropped or imputed), no D feature, C_n_colors kept in the model but not phrased. **Built 2026-10-09: `features/output/feature_table_v1.0.parquet`** (6,743 rows, 42 features, 1 extraction error instead of 85; dictionary lists the QA columns). Source signal unchanged (UFES vs MILK within diagnosis AUROC 0.87–0.93).*

### WS2: One-vs-all tree models, TreeSHAP, top-3 selection (M1 W3 → M2 W8)

- ☑ **T2.1** Define the one-vs-all classes. Use diagnosis classes with enough samples (e.g., melanoma, BCC, SCC, nevus, seborrheic keratosis, other). Decide how to handle "indeterminate" (exclude from training, keep for analysis). Log the decision in §8. *2026-10-01: BCC, Nevus, SCC (invasive + NOS + in situ), SK, Melanoma (invasive + in situ + NOS + metastasis), Other (keratoacanthoma stays in Other).*
- ☑ **T2.2** Train XGBoost/LightGBM one-vs-all models on the S1 split. Use class weights, tune on val only, and report per-class AUROC/F1. *2026-10-02 (v0): XGBoost, val-tuned; one-vs-all test AUROC 0.77 (SCC) to 0.87 (nevus, melanoma); macro-F1 0.44. 2026-10-09 (v1.0, split v2 test, `trees/output_v1.0/`): BCC 0.79, nevus 0.89, SCC 0.69, SK 0.75, melanoma 0.86, other 0.66; macro-F1 0.37.*
- ☑ **T2.3** Compute TreeSHAP (`shap.TreeExplainer`, exact). Choose and document interventional vs. tree-path-dependent. Explain in log-odds space. *2026-10-02: tree_path_dependent, log-odds of malignant, signed toward ResNet50's predicted class.*
- ☑ **T2.4** **Concept grouping:** sum SHAP within each concept group (A, B, C, D, texture, shape, embedding) so correlated features don't fill all three slots. Rank concepts for the **predicted class**. Keep the top-3 with sign and magnitude bucket (strong/moderate/weak). *2026-10-02: grouped top-3 with strength tertiles fixed on val; most-cited groups color 29%, border 24%, size-in-frame 23%.*
- ◐ **T2.5** **Surrogate fidelity (critical).** Measure agreement between the tree prediction and the ResNet50 prediction (→ S1-3). Compare two variants: tree trained on **ground truth** vs. tree trained on **ResNet50's predicted labels** (a true surrogate of ResNet50). Pick based on fidelity (C1), and flag low-agreement cases in the explanation. *2026-10-02 (provisional): surrogate trained on ResNet50 labels = 84.9% agreement (benign 77%, malignant 91%) vs 82.0% for the ground-truth tree, so the surrogate is used; just under C1. 70% agreement on ResNet50-wrong cases. 2026-10-07 (v0.2, split v2 test, n=1,341): 85.5% (benign 74%, malignant 90%), 71% on ResNet50-wrong cases; ResNet50 AUROC without MSKCC is 0.81. 2026-10-09 (v1.0): 85.8% (benign 73%, malignant 91%), 72% on ResNet50-wrong cases. 2026-10-15: refit on the split-v2 deterministic ResNet50 (Narval baseline, lr 3e-4 chosen by val AUROC; ResNet50 test acc 0.832 / AUROC 0.882): surrogate acc 0.784, AUROC 0.811 (was 0.768 / 0.782 on the v1-split model); in `trees/output_v1.0_detv2/`. Agreement with the v2 ResNet50 on test: 84.3% (benign 72%, malignant 89%), 67.6% on ResNet50-wrong cases, lowest on the melanoma class (59.5%); below the v1 refit's 85.8%, still under C1. 2026-10-16: the split-v3 det model (more training data) is harder to imitate: agreement 82.1% (−2.2 points, significant). 2026-10-18: the 72%/89% above are per ResNet call; per true class (C1's definition) v2 is benign 78.9%, malignant 86.6%. `trees/surrogate_sweep.py` (val-only selection over targets, reweighting, fail-mask rows, threshold): best 84.8% (80.2% / 86.7%), so tuning does not reach C1; the gap is in the features. Not adopted yet (it would change every top-3).*
- ☑ **T2.6** Stability: check that the top-3 is stable across seeds and bootstrap resamples (Jaccard of top-3 sets). Report it. *2026-10-02: mean top-3 Jaccard 0.76 across 5 bootstrap refits. 2026-10-09 (v1.0): 0.81. Top-3 groups on test: color 31%, texture 23%, border 22%, asymmetry 16%, shape 8%.*
- ☑ **T2.7** Cross-check with S1's binary tree model and its TreeSHAP (→ S1-5). *2026-10-10 (`trees/compare_drivers.py`): S1's tree matches our GT binary tree (ρ 0.84) and behaves as a non-pigmented-lesion detector (drivers closer to one-vs-all BCC, ρ 0.71, than melanoma, 0.48; darker/browner → benign). Melanomas are as pigmented as nevi, so color pushes them toward benign: recall BCC 93% / SCC 95% / melanoma 26% (one-vs-all 48%, ResNet50 57%). Binary-tree explanations of melanomas would mislead; one-vs-all is the right explanation basis.* *2026-10-09: S1's ground-truth binary tree on v0.2 (`trees/s1_binary_tree.py`, branch `s1-notes-2026-10-07`): test AUROC 0.81, melanoma recall 0.28; driver comparison pending. Re-run on v1.0: AUROC 0.82, melanoma recall 0.26; color dominates its global drivers (mean |SHAP| 1.14 vs ≤ 0.40 for other groups).* Where the malignant-vs-benign drivers disagree with the one-vs-all drivers, document why. *(D2)*

### WS3: Feature→dermatological-concept mapping (M1 W4 → M2 W8)

- ◐ **T3.1** Draft a **concept map and phrase bank**: *2026-10-10: phrase bank v1 (`concepts/phrase_bank.py` → `concepts/output/phrase_bank_v1.csv`) with a status per term: of 34 named terms 3 validated, 8 partial, 4 failed, 19 without usable labels; failed/untested terms reworded as image properties ("red tone", "image texture"); the benign comparison now follows the finding word. Clinician review doc (Claude Doc "Explanation Phrase Bank for Clinical Review") ready for M-6.*
  - one entry per concept group, with graded phrases (e.g., asymmetry: "symmetric / asymmetric in one axis / asymmetric in two axes"; border: "regular / mildly irregular / markedly irregular"; color: "one color / two colors / multicolored, including blue-gray");
  - a definition of each term;
  - its ABCD or clinical relevance.
- ◐ **T3.2** **Calibrate the thresholds with data, not by hand:** *2026-10-07: interim phrase bank v0 = benign-percentile bands (`explain/llm_input.py`); data-calibrated grades pending Derm7pt/PH2.*
  - fit cut-points against labeled datasets: ENHANCE A/B/C scores (ISIC2017 + PH2), PH2 expert asymmetry and colors, Derm7pt criteria, and SLICE-3D metadata (border jaggedness, color variation, `clin_size_long_diam_mm`);
  - where no labels exist, use percentiles of the benign reference distribution.
- ◐ **T3.3** **Validate** each concept (C2): agreement between our computed concept grade and the expert labels. Drop or flag concepts that don't reach the target. Note the domain shift (these sets are dermoscopic; our main set is clinical close-ups). *2026-10-07 (`concepts/`): ENHANCE crowd labels (ISIC 2017, GT masks), each scored against the same mean of other raters: asymmetry features 0.39 vs a human rater 0.38; color 0.34 vs 0.12, but color raters agree below the 0.2 usability threshold, so only asymmetry is validated; border raters agree at 0.01, so border cannot be validated on ENHANCE. Clinical domain, MILK10k MONET scores: pigmentation ρ 0.69; erythema, vessels, ulceration/crust ≤ 0.16, so those clinical words are banned from the phrase bank. Proposal: C2 relative to inter-rater agreement (A-4). 2026-10-09, Derm7pt clinical photos (1,011 cases, expert 7-point labels; masks from the clinical UNet): our features separate expert labels only modestly (AUROC: blue-whitish veil vs C_frac_blue_gray 0.70; elevation vs T_roughness 0.64; pigmentation vs C_color_std 0.63). Qwen2.5-VL as a concept checker: only elevation is usable (32B κ 0.58 vs 7B 0.48); pigmentation, veil and all dermoscopic criteria are near chance (κ ≤ 0.40, mostly < 0.1). The VLM's A/B/C answers do not track our features (asymmetry ρ −0.01, color count 0.11), so they cannot validate C_n_colors. Output: `concepts/output/vlm_20261008_22737290/` (dermoscopy part provisional: 800/1,011 until resume job 22891543).*
- ☐ **T3.4** Embedding features, only if used: build an intermediate mapping from important embedding dimensions to known patterns, for example:
  - probing classifiers trained on Derm7pt criteria, or
  - concept activation vectors (TCAV-style), or
  - WhyLesionCLIP text-concept similarity.
  If none of these is reliable, label these features "learned visual pattern (not clinically mapped)" instead of inventing a concept.
- ◐ **T3.5** Send the concept map and phrase bank for clinical review (→ M-6). The prior texture definitions were never checked by a clinician. *(D3)* *2026-10-18: review package ready in English and Quebec French: `explain/lang_fr.py` (French twin of the template layer, draft), `concepts/avoided_terms.py` (15 avoided-term entries with reason, evidence, preferred wording and EN/FR stems; `hits()` checker, clean on 592 real Qwen explanations) -> `concepts/output/clinical_review_v1_en_fr.md`. French is not shown to anyone before the reviewer approves it.*

### WS4: LLM setup and input structuring (M1 W3 → M2 W6)

- ◐ **T4.1** Shortlist models: *2026-10-07: Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct (both Apache-2.0, ungated) served with vLLM on Rorqual H100 (since 2026-10-14 on Narval A100: Rorqual's queue for our account stalled, see §8); Qwen3-32B needs 2 A100s (`VLLM_TP=2`); Llama-3.1-8B optional (gated licence).* Llama 3.x 8B Instruct and Mistral 7B Instruct (grant), plus optionally a biomedical-tuned variant. Check the licenses for commercial use (→ M-8). Serve locally on GCP (vLLM or HF transformers) so no data leaves MedirAI infrastructure.
- ☑ **T4.2** Define the **input JSON schema**. *2026-10-07: `explain/llm_input.py` (`llm_input_v1`): number-free; findings, benign-relative bands, confidence category, uncertainty flag, surrogate agreement, segmentation reliability; S_hu moments marked not clinically mapped.* It contains:
  - the prediction and probability;
  - the uncertainty/deferral status and reason (from S1-4, e.g., "models disagreed", "high entropy");
  - surrogate agreement;
  - the top-3 concepts, each with phrase, direction ("supports malignant" / "argues against"), strength, and `feature_id`;
  - the target audience.
  It must never contain images, metadata, or identifiers.
- ☑ **T4.3** Define the **output JSON schema**: *2026-10-07: `OUTPUT_SCHEMA`, enforced with vLLM JSON-schema decoding.* `summary`, `reasons[] {text, feature_id, direction}`, `uncertainty_note`, `limitations`. Add constrained decoding or a JSON grammar if the serving stack supports it.
- ☑ **T4.4** Build a **template baseline (no LLM)** that turns the same JSON into fixed sentences. It serves as the lower bound and the fallback. *2026-10-02: `explain/template.py`; every reason cites a feature_id; uncertainty, surrogate disagreement and mixed evidence always stated.*

### WS5: Prompting strategies (M2 W7 → M3 W10)

- ◐ **T5.1** Implement strategies: *2026-10-07: (a) `zs_v1` and (c) `cite_v1` in `explain/prompts/`; (b) few-shot and (d) GP variant pending.*
  - **(a)** zero-shot structured prompt;
  - **(b)** few-shot, with care, because examples were copied verbatim in prior work;
  - **(c)** "cite-your-feature" prompt requiring a `feature_id` per claim;
  - **(d)** audience variants (dermatologist vs. GP; confirm the target with M-5).
- ◐ **T5.2** Protocol, following "Reproducible Prompt Testing": *2026-10-07: greedy decoding, seed 0; every run logs model + revision, prompt version + hash, decoding, input hashes (`explain/llm_generate.py`).*
  - explore with fixed-seed, low-temperature sampling or beams;
  - evaluate with fully deterministic decoding;
  - fixed eval set;
  - log model id, prompt version, decoding params, and input hash for every run.
- ☐ **T5.3** Guardrail instructions to test:
  - mention only the supplied features;
  - no diagnosis beyond the model's output;
  - state uncertainty or deferral first when present;
  - say plainly when evidence is mixed;
  - no treatment advice.
- ◐ **T5.4** Explicit tests on **wrong** and **deferred** predictions (→ S1-9) so explanations don't lend false confidence. *(D4, prompt library)* *2026-10-02: done for the template and Grad-CAM (LLM pending). Confidently wrong predictions get explanations indistinguishable from confidently correct ones (surrogate agrees 92% vs 92%; no caution 62% vs 63%); only the uncertainty flag separates them. 2026-10-10 (`eval/error_signals.py`): among confident predictions, the ground-truth feature tree's doubt flags ResNet50 errors better than entropy (test AUROC 0.66 [0.62–0.71] vs 0.61); new code-built caveat (`llm_input_v2` `independent_check_weak`, cut fit on val) appears on 40% of confident-wrong vs 19% of confident-correct cases; confident wording wrong vs correct 48% vs 68% (was 61% vs 76% without it).*

### WS6: Automated validation (M3 W9 → M4 W14)

- ◐ **T6.1** **Verifier** (rule-based + NLI): *2026-10-07: rule-based part done (`explain/verifier.py`, negative-control tests); template arm passes 100% on 1,341 test cases. 2026-10-10: NLI claim check (`explain/nli_check.py`, DeBERTa-v3-large NLI) detects 100% of flipped findings and directions and 99.3% of swapped measures with 0% false alarms on synthetic controls; false-alarm rate on real LLM paraphrases still to measure. Verifier now allows numbers copied from raw-number encodings. On the 2026-10-08 hybrid_v1 runs, NLI flags 0.3% of claims (Qwen3-32B), 3.9% (Qwen2.5-7B), 11.3% (Mistral-7B); it is complementary to the rule verifier (Qwen2.5-7B: 72 explanations flagged only by NLI, 29 by both, 10 only by rules). A sample of flags shows real errors (direction flips, wrong magnitude, invented justifications) and some false alarms on filler phrases; precision needs the human labels (RQ1).* grounding rate (C3), top-3 coverage, direction correctness (C4), invented-concept detection against the concept vocabulary, and the uncertainty-flag rule (C5).
- ◐ **T6.2** **Counterfactual sensitivity:** *2026-10-10: `explain/counterfactuals.py` (flip/swap/drop/repeat, 75 cases per stratum); template scores 1.0 (scorer sanity), negative-control tests; LLM runs in `slurm/rorqual_llm_d5.sh`. 2026-10-16: submitted on Narval (5852520) on the v2 deferral inputs; done, results in RESULTS.md (D5 section). Qwen3-32B keeps untouched claims stable far more often than Qwen2.5-7B.* perturb or swap input concepts, then check that the explanation changes accordingly and unchanged parts stay stable.
- ◐ **T6.3** **Consistency:** *2026-10-10: `explain/consistency.py` (same-signature similarity, paraphrase `hybrid_v1p`), determinism via repeat variants; runs in `slurm/rorqual_llm_d5.sh`. 2026-10-16: run on Narval (5852520). Repeats are 99% identical, not 100%, under `VLLM_TP=2`, so tensor-parallel runs are not fully deterministic.*
  - similar inputs (same top-3 concepts and directions) should produce semantically similar outputs;
  - outputs should be robust to prompt paraphrase;
  - the same input run repeatedly should give identical output.
- ◐ **T6.4** **LLM-as-judge:** *2026-10-18: `explain/llm_judge.py` (claim labels supported / direction / unsupported concept / magnitude / other, plus uncertainty, overconfidence, clarity, tone; planted-error controls for recall and specificity; blinded 60-case rating sheet; judge-vs-human κ). Judges Gemma-3-27B-it and Llama-3.3-70B AWQ on the D5 base cases (`slurm/narval_llm_judge.sh`; Llama job 6206187 queued, Gemma waits for an HF token). Llama (6226589): planted-error recall 0.98–1.00 per type, specificity 0.79 on clean copies; most false alarms are benign-typical values that support malignant, which the judge overrides with its own clinical prior. Direction labels from the judge are therefore not used (consistent with H3).* *2026-10-21: Gemma rerun done (§8); human calibration (Michael's ratings) pending.*
  - use a different model family from the generator;
  - rubric: accuracy vs. input, coherence, clarity, clinical tone;
  - **calibrate the judge against human ratings** on ~50 samples (Spearman / κ) before trusting it at scale.
- ◐ **T6.5** **Stratified sampling:** correct vs. incorrect × confident vs. deferred × class. Report every metric per stratum, including C6 (confident language on wrong cases). *2026-10-02: strata implemented in `eval/wrong_prediction_study.py`; error-detection AUROC: model entropy 0.74, explanation support 0.65, Grad-CAM 0.54.*
- ◐ **T6.6** Compare: *2026-10-07: `explain/compare_runs.py`, run per job by `slurm/rorqual_llm.sh`.* template baseline vs. each prompt strategy vs. each LLM. Report latency and GPU memory per explanation. *(D5)*

### WS7: Clinician review (prep M1–M2, run M3, analyze M4)

- ☑ **T7.1** Ethics determination and approval **before any clinician-facing session** (→ A-1). *Answered 2026-10-09: Prof. Abusitta confirmed no Polytechnique ethics certification is required. Clinician recruitment (M-7) is now the critical path.*
- ◐ **T7.2** Study design (draft end of M2). Proposal: *2026-10-02: draft 40-case set (5 per correct/wrong × confident/uncertain × benign/malignant cell) with label-only, Grad-CAM and template materials and a separate answer key. 2026-10-18: request to clinicians drafted (one clinical reviewer, ~2 h, by Nov 13; ≥ 3 raters, ~75 min, Nov 30–Dec 13); the reviewer is also asked how to present counter-intuitive reasons (a benign-typical value supporting malignant), which the LLM judge flagged.*
  - **Case set:** ~40 cases stratified as in T6.5, frozen model version (→ S1-8).
  - **Sequential reveal per case:**
    1. The clinician's own impression, unaided.
    2. The model label and probability.
    3. Plus Grad-CAM *or* plus the NL explanation. Arm order is counterbalanced across clinicians.
  - **Ratings (5-point):** clinical relevance, trustworthiness, clarity. Also: "does this support or change your judgment?" and free text.
  - Include wrong-prediction cases to measure automation bias.
- ◐ **T7.3** Materials (2026-10-18: Poly-Opinion, Polytechnique's LimeSurvey, chosen over a custom platform; `eval/clinician_study/build_survey.py` builds the stratified case set, per-rater counterbalancing, answer key, bilingual LimeSurvey import file and an offline HTML preview; 5-case pilot in `eval/output/clinician_study_pilot/`; import not yet tested in Poly-Opinion): rating form (Google Form or a simple web page), Grad-CAM images from S1's ResNet50 (→ S1-7), and a clinician instruction sheet. *2026-10-02: Grad-CAM arm built on the predicted class (P1-16); deletion AUC 0.656 vs 0.641 for random order, i.e. not faithful; 33% of CAM mass on the lesion (22% of the image).*
- ☐ **T7.4** Recruit and schedule clinicians (→ M-7). Target ≥ 3 dermatologists; hold sessions in M3 W11–12.
- ☐ **T7.5** Analysis: descriptive statistics per arm and stratum, inter-rater agreement, qualitative themes. Feed the findings back into the prompts and concept map. *(D6)*

### WS9: Dataset development (added 2026-10-02; MedirAI has no images yet, see M-4)

- ◐ **T9.1** Assess the current data. *2026-10-02: 7,983 labelled ISIC clinical close-ups (7,582 lesions). Malignant is 71% BCC; melanoma 522 images (348 train); few benign mimics (SK 575, LPLK 135, solar lentigo 63, dermatofibroma 52); only 1,378 benign images biopsy-confirmed; Fitzpatrick V–VI ≈ 5% of the 2,457 with a recorded type. Learning curve (25/50/100% of training groups, 2 runs each): test accuracy 0.805 → 0.830 → 0.840, AUROC excluding MSKCC 0.788 → 0.805 → 0.836, i.e. about +2 points per doubling, not yet flat. BCC/SCC are saturated (~0.97); nevus and SK keep improving; melanoma falls (0.63 → 0.54) as more non-melanoma data is added, so melanoma needs targeted data and class weighting, not just volume. Train ≈ val accuracy at every size, so the current recipe (lr 1e-5, 11 epochs) under-fits; tune it before concluding how much data is needed. 2026-10-07: the 1,240 MSKCC images (73 patients, 97% unbiopsied benign, dermoscopy-like contact images, many without a visible lesion) are excluded in `splits/isic_clinical_v2.csv` (v1 assignments unchanged; licence and label-confirmation columns added). v2 = 6,743 images, 1,962 benign.*
- ◐ **T9.2** Check candidate public sources *2026-10-07: PAD-UFES-20 and MILK10k are already in our data. Priority: MRA-MIDAS (prospective, biopsied, iPhone 15/30 cm; needs Redivis account), Derm7pt (252 melanoma; form ready), DDI (fairness test set). MED-NODE downloaded (no licence stated; external test only). PH2 official link dead. SLICE-3D/Fitzpatrick17k not for training. 2026-10-16: second sweep (SkinCon, ISIC clinical overview, PASSION, SCIN, DermaCon-IN, Derm1M, MoleMapper, MCR-SL, synthetic sets incl. cgDDI). New usable: SkinCon dermatologist concept labels on 910 neoplastic Fitzpatrick17k photos (WS3) and 432 ISIC clinical-overview images (284 malignant, 103 melanoma, CC-BY/CC-0, no lesion overlap with v2). 376 of ISIC's 387 Fitzpatrick V–VI close-ups are MSKCC (excluded), so dark-skin cancer data exists publicly only in DDI. cgDDI is built from DDI images, so it can't be used while DDI is our test set.* *2026-10-20: Redivis access granted (exports unrestricted) to MRA-MIDAS (3,416 images, biopsied, phone + dermoscopy, mm lesion size), DDI (656) and DDI-2 (665); download to Narval pending a Redivis API token.* for licence, label provenance and overlap: MILK10k and PAD-UFES-20 (likely overlap with our MILK/UFES images), Derm7pt (melanoma-rich; also WS3), DDI (biopsy-proven, balanced skin tones; fairness test set), MED-NODE, Dermofit (paid; has lesion masks), Fitzpatrick17k and SLICE-3D (secondary). Dermoscopic sets only for pretraining.
- ☐ **T9.3** De-duplicate every new source against the current data by lesion and patient (and near-duplicate image hashing) before use; new sources go into a new split version (`isic_clinical_v2`), never by re-splitting v1.
- ◐ **T9.5** Skin-tone reporting (added 2026-10-16): `eval/skin_tone_report.py` (MILK10k tone 0 = darkest, ISIC Fitzpatrick). Test has 120 darker-tone images (MILK 0–2 / FST V–VI); v2 det AUROC darker 0.805 vs lighter 0.907, v3 det 0.840 vs 0.927 (malignant share differs: 0.44 vs 0.79).
- ◐ **T9.6** External test sets (added 2026-10-18): MCR-SL (Zenodo 17306338, md5 checked; originals on Narval `$SCRATCH/medirai/external/mcr_sl`, ≤1024-px copy in `data/mcr_sl/MCR-SL_1024`), ISIC clinical overview (432/432 at full resolution in `data/isic_clinical_overview/images`), Asan/SNU (406), SNU quiz (240, not extracted). Not yet cropped, de-duplicated or scored. *2026-10-21: scored v2/v3 det x 2 seeds on ISIC overview, MCR-SL, DDI, DDI-2, MIDAS (`eval/external_test.py`); now locked test sets (§8).*
- ☐ **T9.4** Datasheet for the assembled dataset: sources, licences, label provenance (histopathology vs clinical), class and skin-type balance, known confounds.

### WS8: Integration, documentation, reporting (M3 W11 → M4 W16)

- ◐ **T8.1** Package it as a module: `explain(isic_id | features, prediction, uncertainty) → {top_concepts, text, verification}`. Agree the integration target with MedirAI (→ M-10), e.g., a new endpoint in the payload-generation FastAPI. *2026-10-21: `explain/module.py` (one call per case, verifier-gated, GP glossary, French template); API target still M-10.*
- ☐ **T8.2** Integrated prototype with S1's refined ResNet50 (★ End M3).
- ☐ **T8.3** Re-run the full pipeline and evaluation on S1's final model (→ S1-8).
- ☐ **T8.4** Documentation: README, feature dictionary, concept map, prompt library, eval protocol, known limitations (no mm diameter, no "E", clinical-vs-dermoscopic domain shift, surrogate-fidelity caveat).
- ☐ **T8.5** Privacy/data-use summary contribution (*D8*), joint report with S1, Mitacs final report, exit survey.

---

## 6. Dependencies on Student 1, MedirAI, and the supervisor/admin

### 6.1 From Student 1 (Shatha)

| ID | What I need | Needed by | Blocks | Status |
|---|---|---|---|---|
| S1-1 | Final train/val/test split file for ISIC clinical close-ups (70/10/20, **grouped by lesion/patient**, keyed by `isic_id`, all diagnosis levels). Produced by CODEBASE_TODO P0-1/P0-2 (co-owned) | M1 W2 (Oct 4) | T0.6, T0.8, T1.5, T2.2 | ☑ `splits/isic_clinical_v1.csv` (built by S2, 2026-09-25); superseded by `isic_clinical_v2.csv` (2026-10-07), which S1 adopted in `configs/medir_v2.json` |
| S1-2 | Co-design of the feature schema and extraction code; agreement on mask source and quality rules | M1 W3–4 | WS1 | ◐ S1 reviewed v0.2 (2026-10-08): 4 questions (failed rows, mask_quality, diameter, C_n_colors) before the v1 freeze |
| S1-3 | ResNet50 checkpoint **retrained on `isic_clinical_v1` after the P0 fixes**, + per-image predictions/probabilities on val/test (CSV by `isic_id`). A preliminary version is fine at first; final at the End-M2 milestone | M2 W5 (prelim), W8 (final) | T2.5, T4.2 | ◐ Preliminary done 2026-10-15 (v2 det, lr 3e-4, Narval); v3 det + second seeds 2026-10-19. Final = the model chosen with S1 (decision due Oct 23) |
| S1-4 | Per-image uncertainty outputs: entropy-of-expected, expected entropy, variational variance, multi-model agreement/deferral flag, and which models disagreed | M2 W8 | T4.2, T6.5 | ☑ 2026-10-15: per-image uncertainty table from the v2 baseline; deferral flag wired into the explanation inputs (`explain/attach_deferral.py`). Redo on the chosen model |
| S1-5 | Binary tree model + its TreeSHAP values (for cross-checking against one-vs-all) | M2 W8 | T2.7 | ◐ Done on v0.2 by S1; re-run on v1.0 by S2 (2026-10-09, `trees/output/s1_binary_v1.0/`); S1 to review |
| S1-6 | ResNet50 penultimate-layer embeddings on all splits, only if embeddings are used as features | M2 W6 | T1.3 (opt.), T3.4 | ☐ |
| S1-7 | Grad-CAM heatmaps from the ResNet50 for the frozen clinician-review case set | M3 W10 | T7.3 | ☐ |
| S1-8 | Refined ResNet50 (versioned) + re-exported predictions/uncertainty; one version frozen for the clinician study | M3 W10 (freeze), M4 W13 (final) | T7.2, T8.2, T8.3 | ☐ |
| S1-9 | Lists of misclassified and uncertain/deferred cases + S1's attribution findings on them | M3 W9 | T5.4, T6.5 | ☐ S1 sends them once the v2 predictions exist |

**What I provide to Student 1:**
- the feature table and dictionary (W5–6);
- the concept map (W8);
- one-vs-all SHAP outputs (W8);
- NL explanations for deferred and disagreeing cases (M3). The grant wants deferred predictions to come with an interpretable rationale.

### 6.2 From MedirAI

| ID | What I need | Needed by | Blocks | Status |
|---|---|---|---|---|
| M-1 | Access to `medirai_payload_generation_api`, the payload_generation fine-tuning/segmentation repo, `medirai_malignant_prediction_api` (`dev_chloe`), and GCS `medirai-storage-bucket-medirai-production/payload_generation_models` | M1 W1 | T0.1, T1.3 | ◐ Requested 2026-09-25 |
| M-2 | GCP VM with GPU (an 8B LLM in bf16 needs ~16–20 GB VRAM; A100 is fine) + working SSH/VS Code (contact: Pasindu) | M1 W2 | T0.2, WS4–6 | ◐ Requested 2026-09-25 |
| M-3 | Segmentation model weights and/or pseudo-masks for ISIC clinical close-ups (Aloys's UNet / DeepLabV3+-ResNet101) | M1 W3 | T1.2 | ◐ Requested 2026-09-25 |
| M-4 | Decision on whether proprietary MedirAI clinical images are in scope. If yes: data-use agreement, access path, de-identification confirmation | M1 W2 | T1.5, A-1 | ☑ **Answered 2026-10-02: MedirAI has no clinical images of its own yet.** S1 and S2 build the training and evaluation dataset from public sources (WS9) |
| M-5 | Explanation spec: target reader (dermatologist vs. GP; prior prompts targeted GPs), length, language (EN/FR?), where it appears in the product | M1 W2 | T4.2, T5.1 | ◐ 2026-10-21: dermatologists first, GPs second. Length, language and product placement still open |
| M-6 | Clinical reviewer (in-house advisor or contact) to sanity-check the concept map, phrase bank, and definitions | M2 W7 | T3.5 | ☐ |
| M-7 | Recruit ≥ 3 dermatologists for the review; confirm availability for M3 W11–12 and any compensation or logistics | Confirmed by M2 W8 | T7.4 | ☐ |
| M-8 | LLM licensing/deployment constraints: Llama 3 Community License vs. Mistral (Apache-2.0); on-prem only? | M1 W2 | T4.1 | ☐ |
| M-9 | Sprint check-in cadence (weekly or biweekly), shared repos/Drive, who attends | M1 W1 | T0.4 | ☐ |
| M-10 | Integration target and interface for the module (e.g., payload API endpoint), plus handoff format | M3 W11 | T8.1 | ☐ |

### 6.3 From the academic supervisor / administration

| ID | What is needed | Needed by | Blocks | Status |
|---|---|---|---|---|
| A-1 | Ask Polytechnique's research ethics office (CER) whether certification is required for (a) secondary use of ISIC and possibly MedirAI images and (b) the clinician review. Submit if required. **The grant §3.2 declarations currently answer "No" to human participants and secondary health data, while §2.2 plans a clinician review. Reconcile this.** | Started M1 W1; cleared by M3 W10 | WS7 (critical path) | ☑ 2026-10-09: Prof. Abusitta confirmed no Polytechnique ethics certification is required |
| A-2 | Mitacs Part 2 completeness. Confirm approval status (Mitacs requires approval before the project begins). Open gaps in the PDF: Intern 1 details (§4.3.1), Section 5 suggested reviewers (empty), ORS signature (§7.5), Faculty field (§4.2.1), supervisor name typo "Adel Aby Sitta" (§7.2), duplicated Month 4 Gantt row for Student 2 | ASAP | Funding, timeline | ☐ |
| A-3 | IP: extension of the existing research agreement (§3.9) | Before handoff (M4) | T8.1 | ☐ |
| A-4 | Agree on the success criteria (§1) and the clinician-study design (T7.2) | M1 W2 / M2 W8 | T0.4, T7.2 | ☐ |

---

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tree model has low fidelity to ResNet50, so the explanation describes a different model | Explanations are unfaithful | T2.5: surrogate-on-ResNet50-labels variant, report C1, flag disagreement cases in the text |
| Handcrafted features depend on segmentation, and masks fail on clinical close-ups | Wrong concepts | Mask-quality flag (T1.2), now checked against the gold set (bad-mask rate 12% ok / 91% fail); abstain from affected concepts; say "segmentation uncertain" |
| Concept thresholds don't match clinical judgment (prior color/texture rules ~27–30%) | Misleading phrases | Data-calibrated thresholds + validation (T3.2–3.3); drop weak concepts; clinical review (M-6) |
| Dermoscopic validation sets vs. clinical close-up target | Validation doesn't transfer | Report the domain shift; validate on SLICE-3D/PAD-UFES clinical metadata where possible |
| LLM invents features or overstates confidence (confirmation bias seen in prior XAI work) | Unsafe output | Structured output + verifier (T6.1); tests on wrong/deferred cases (T5.4); template fallback |
| Clinician recruitment slips (ethics cleared 2026-10-09: not required) | D6 late | Recruit via M-7 by W8; fallback: pilot with fewer clinicians + stronger automated eval |
| Upstream delays from S1 (model, uncertainty outputs) | WS2/WS4 blocked | Build against a preliminary S1 checkpoint on the new split. Don't use the NRC ensemble until its P0 items are fixed. Keep interfaces keyed by `isic_id` so the swap is trivial |
| Legacy accuracy numbers are unreliable (leakage, source confound, broken heads; see CODEBASE_TODO P0) | Wrong baselines in the report; C1 fidelity measured against a bad model | Fix P0 in W2–W3, re-baseline once, and cite only post-fix numbers. Label any legacy number as "pre-fix" |
| Diameter has no mm calibration; "E" is not observable | Incomplete ABCDE | Use `clin_size_long_diam_mm` (1,229 images, mostly UFES) to validate or calibrate a pixel-Feret proxy; elsewhere, pixel diameter with a caveat. State that E is out of scope in every explanation's limitations |
| Label/source confound in the public data: MSKCC images are 99.5% benign, mostly unbiopsied ("single contributor clinical assessment"), and the only >2k-px photos; only 1,378 of 3,196 benign images are histopathology-confirmed | Models and features learn the source or verification status instead of the lesion | Report every metric per source and with MSKCC excluded; features at a common working size; benign-only source-shortcut check (target AUROC ≈ 0.5); prefer biopsy-confirmed benign data in WS9 |
| No MedirAI images yet (M-4), so the target domain is approximated by public clinical photos | Results may not transfer to MedirAI's deployment | WS9: assemble and document a public clinical dataset; keep the pipeline data-agnostic so MedirAI data can be added later |
| Class imbalance for rare one-vs-all classes | Unstable SHAP | Merge rare classes into "other"; class weights; stability check (T2.6) |
| GPU memory and repo access issues (both happened before) | Lost time | Request access in W1 (M-1, M-2); 8B models only; quantize if needed |
| Llama license unsuitable for commercial use | Rework | Decide early (M-8); keep Mistral (Apache-2.0) as a drop-in alternative |

---

## 8. Open questions and decision log

| Date | Question / decision | Owner | Outcome |
|---|---|---|---|
| 2026-09-25 | Feature layer ownership | S1+S2 | **Decided:** shared/co-owned. S1 trains the binary tree; S2 trains one-vs-all |
| 2026-09-25 | Start date of Month 1 | — | **Decided:** Mon 2026-09-21; end Jan 20, 2027 |
| 2026-09-25 | Codebase layout | S2 | **Decided:** one git repo at the workspace root. Duplicates removed, superseded code in `poc-nrc-main/archive/` |
| 2026-10-02 | Which baseline number is cited going forward | S1+S2 | Provisional (local): deterministic ResNet50 0.828 test accuracy, AUROC 0.884; variational 0.837–0.840. Official = S1's cluster re-run of the same scripts. Always also report per source |
| 2026-09-25 | Handling of "indeterminate" | S2 | **Decided:** Indeterminate and missing `diagnosis_1` are `split=excluded` (1,333 images). They're kept for analysis only |
| 2026-10-01 | One-vs-all class set | S2 | **Decided:** BCC, Nevus, SCC, SK, Melanoma, Other (keratoacanthoma in Other) |
| 2026-10-02 | Tree trained on ground truth vs. ResNet50 labels | S2+S1 | **Decided (v0):** ResNet50-label surrogate (fidelity 84.9% vs 82.0%) |
| 2026-10-01 | Lesion masks while M-3 is pending | S2 | **Decided:** own ResNet18-UNet on ISIC 2018 + zero-shot SAM agreement flag; masks not convex-hulled |
| 2026-10-02 | Variational-layer loss (P1-4) | S1+S2 | Both objectives tested; results tie, and the posterior σ never moves at lr 1e-5 (P1-5). For S1 to decide |
| 2026-10-02 | Explained model | S2 | Deterministic ResNet50 for Grad-CAM, surrogate fidelity and C6; variational results reported alongside |
| 2026-10-07 | MSKCC images | S2 (+S1 to confirm) | **Decided:** excluded in split v2 (different modality and population); S1 re-baselines on v2 |
| 2026-10-07 | Licence policy for new data | S2 | **Decided:** research/NC sources allowed; licence recorded per image; commercial-safe subset for MedirAI |
| 2026-10-07 | Feature version | S2 | **Decided:** v0.2 (color constancy, lesion-scale texture, no frame-relative size in the model) |
| 2026-10-07 | Explanation audience | S2 | **Decided (until M-5):** dermatologist, English; `audience` field kept for GP/French |
| 2026-10-07 | C2 target | S2 → A-4 | **Proposed:** concept features must agree with the rater consensus at least as well as a single rater does; absolute ρ ≥ 0.5 is unreachable on non-expert labels (border raters agree at 0.01) |
| — | Use ResNet50 embeddings as features? | S2+S1 | Open: only if the intermediate concept mapping is reliable (T3.4) |
| 2026-10-21 | Target reader of explanations (M-5) | M | **Answered:** dermatologists primary, GPs secondary. **Decided (S2):** same verified reasons for both; GPs additionally get a code-built plain-language glossary of the cited concepts (`explain/module.py`, draft for the reviewer), so the LLM step and its evaluation are the same for both audiences. Language still open |
| 2026-10-02 | Proprietary MedirAI data in scope? | M | **Answered:** MedirAI has no images yet. We assemble the dataset ourselves from public sources (WS9) |
| 2026-10-09 | Ethics certification required? | A | **Answered:** not required (Prof. Abusitta). Keep the confirmation for D8 |
| 2026-10-09 | Gold mask set | S1 | **Done:** 97/100 traced by S1 blind to model masks; used to choose the mask source and check the flag (T1.2) |
| 2026-10-09 | `bayeslayers.py` / `ConvTranspose2dReparameterization` (P0-17) | S1 | **Answered:** S1 doesn't have it. Only the Bayesian UNet needs it, and nothing in either stream uses that model |
| 2026-10-09 | Mask source for feature table v1 | S2 (+S1 to agree) | **Decided:** clinical fine-tuned UNet (`unet_r18_clinical`), chosen on the gold set (Dice 0.71 vs 0.61). Refit the ok/review/fail thresholds on the gold tune half and quote numbers from the report half. This resolves the 85 empty-mask rows without imputation. Rebuild the table, then retrain the one-vs-all and binary trees |
| 2026-10-09 | Diameter in the model | S2 | **Decided:** no D feature. Pixel size depends on framing, and mm diameter exists only for UFES (missingness reveals the source). Explanations state that size is not assessed from a single photo; `frame_*` columns stay QA-only |
| 2026-10-09 | `C_n_colors` | S2 | **Decided:** kept as a tree feature, but banned from the phrase bank (no "N colors" wording) until it's validated against Derm7pt/PH2 expert color labels |
| 2026-10-09 | Who runs the v2 baseline (`slurm/rorqual_baseline_v2.sh`) | S2 | **Decided:** S2 submits on Rorqual (def-omar12); S1's branch merged locally |
| 2026-10-10 | Independent-check caveat | S2 | **Decided:** confident predictions whose GT feature tree doubts them get a code-built caution sentence (llm_input_v2); cut refit on val whenever the explained model changes |
| 2026-10-14 | Cluster for GPU jobs | S2 | **Decided:** move from Rorqual to Narval (same allocation, `def-omar12_gpu`). Rorqual jobs sat PENDING for six days (account fair-share 0.00004); Narval's was 1.7. Narval has A100-40GB only, so 32B models run tensor-parallel on 2 GPUs. Rorqual maintenance window 2026-10-27 to 11-03 |
| 2026-10-15 | Baseline v2 learning rate | S2 | **Decided:** lr 3e-4 (val AUROC det .896 / vll_std .893 / vll_elbo .903 vs .893 / .882 / .898 at 1e-4; test AUROC det .882). `Val Sigma` moves (0.0499 to 0.052), so P1-5 is closed. The v2 det checkpoint at lr 3e-4 is the explained model |
| 2026-10-15 | What counts as "uncertain" in explanations | S2 | **Decided:** the per-image deferral flag from the uncertainty table (val-p75 entropy OR disagreement between the deterministic and variational heads; 32.4% of the test set) replaces the entropy threshold alone (S1-4). `explain/attach_deferral.py` writes the flag into the inputs; without it the entropy rule still applies |
| 2026-10-10 | Phrase bank | S2 | **Decided:** v1 keeps unvalidated terms in neutral image-property wording, labels each validated / partial / failed / untestable / untested, and leaves the final call to the clinical reviewer (M-6) |
| 2026-10-10 | S2 paper framing and pre-registration | S2 | **Decided:** after a novelty check (Kim et al. 2026 already shows raw→descriptor encoding gains), the lead contribution is the false-reassurance mitigation (H5) and the validated claim-level protocol; H1–H2 become confirmatory. Pre-registration frozen before the v2-model runs: `S2_PAPER_PREREG_2026-10-10.md`, SHA-256 726a0be84f04a5e9b5ded81d1aba4ba7fcafa254021d0fa3322d3495f5feacaa |
| 2026-10-16 | Explained model after the split-v3 and 384-px runs | S2+S1 | **Open:** v3 det vs v2 det (paired bootstrap): melanoma recall +0.13 [+0.04, +0.22], AUROC +0.014 (n.s.), surrogate agreement −0.022 (significant). Proposed: v3 det for the clinician-study freeze if a second seed holds; current LLM/D5 runs stay on v2 det |
| 2026-10-16 | Skin-tone reporting | S2 | **Decided:** every classifier and explanation metric is also reported by tone group (darker = MILK10k 0–2 or FST V–VI). The two scales are kept separate in the table |
| 2026-10-16 | Synthetic images | S2 | **Proposed:** train-only augmentation or label-free counterfactual tests (same lesion, different surrounding skin), never in val/test or concept calibration; cgDDI excluded while DDI is the fairness test set |
| 2026-10-18 | LLM judges (prereg: chosen before human labelling) | S2 | **Decided:** google/gemma-3-27b-it and casperhansen/llama-3.3-70b-instruct-awq (rev 64d2556, 4-bit, Llama 3.3 licence), both other families than the Qwen/Mistral generators; report their agreement with each other and with the human labels. Evaluation-only use, so M-8 licensing applies to the generators, not the judges |
| 2026-10-18 | Seed variance for the explained-model choice | S2 | **Decided:** second seed for v3 det (6205251) and for v2 (6205259, all three heads) before choosing; Narval start estimate Oct 19 |
| 2026-10-19 | Second seeds in (explained model, S1-8) | S2+S1 | **Evidence, decision still Oct 23:** v3 beats v2 in both seeds (paired bootstrap, seed 0 / seed 1): melanoma recall +0.13* / +0.10 (CI touches 0), AUROC +0.014 / +0.031*, surrogate agreement −0.022* / −0.028*. Seed-to-seed spread within a recipe: melanoma recall up to 0.08, AUROC ≤ 0.011. S2 recommends v3 det. Clinical-weighted v3 variant (`--clinical_share`, S1's idea) submitted 2026-10-20 (6551696) |
| 2026-10-21 | Clinical-weighted v3 (`--clinical_share 0.33`, 6551696) | S2 | **Evidence:** no measurable change vs v3 seed 0 (paired bootstrap, every CI includes 0: accuracy +0.008, AUROC +0.004, melanoma recall −0.026, surrogate agreement 0.000); uncalibrated ECE worse (0.114 vs 0.066). Not adopted; v3 det (plain shuffle) stays the recommendation for Oct 23 |
| 2026-10-21 | External tests of v2/v3 det (2 seeds each; `eval/external_test.py`, `eval/output/external/`) and C1 by feature set | S2 | **Evidence:** every model loses much of its discrimination off-distribution. AUROC internal 0.87-0.90 → ISIC overview 0.79-0.86, MCR-SL 0.75-0.83, MIDAS phone photos 0.63-0.69, DDI 0.61-0.71 (near chance for FST V-VI, 0.59-0.61), DDI-2 0.59-0.69. v3's internal gain does NOT transfer: v3 is not better than v2 on DDI/MIDAS/ISIC overview (v2 s0 is best on 3 of 5), and seed-to-seed spread is as large as the v2-v3 gap. Entropy flags errors on shifted data less well (0.48-0.62 on DDI/MIDAS vs 0.71-0.77 internal). **C1:** adding 12 named MONET concept scores to the 42 handcrafted features gives 85.2% (vs 84.8%); adding 16 PCs of the ResNet's own embedding gives 94-95% but those inputs have no clinical names. Plan: explained model decision on Oct 23 must weigh this (recommendation changes from 'v3' to 'v3 not justified on external data') |
| 2026-10-21 | Classifier sprint (S1 stream, S2 proposal) and its evaluation protocol | S2 (Michael approved) → S1, A-4 | **Decided:** rank candidates on external generalization, not internal test. ResNet50 stays the product model; foundation models only as teachers (DINOv2, Apache-2.0) or baselines (PanDerm, CC-BY-NC-ND). MIDAS is training/selection data (patient-grouped `splits/midas_v1.csv`, 60/20/20 train/dev_val/dev_test); DDI, DDI-2, MCR-SL and ISIC overview are locked (scored only with `eval/external_test.py --final`, every look logged). Selection score = mean(internal val AUROC, MIDAS dev_val AUROC) over 3 seeds; targets K1–K9. Pre-registration frozen: SHA-256 6af996b7c6f3c3126e4e85663428af421e3ce1661c430b79f035b1a95c969286 |
| 2026-10-21 | Classifier sprint stage A (selection data only) | S2 | **Evidence:** 8-view TTA adopted (v2 selection score 0.794 → 0.806); crops and padding at test time do not help. A frozen DINOv2 linear probe matches the fine-tuned ResNet50; adding MIDAS train lifts phone-photo AUROC 0.70 → 0.73. Error audit: on phone photos the ResNet flags red/scaly/keratotic benign lesions as malignant (seborrheic keratosis 53% correct) and misses flat cancers (BCC recall 0.65). Stage B order: MIDAS train data, diagnosis-balanced sampling + 7-class head, phone-photo lesion localization, then resolution and augmentation |
| 2026-10-21 | Interim explained model | S2 (S1 to confirm) | **Decided (interim):** v2 det seed 0 with 8-view TTA (best on the external sets; all LLM/D5/C6 results already on it). The sprint's final model replaces it at the Nov 23–29 freeze |
| 2026-10-21 | Stage B scope | Michael | **Decided:** split v4 adds MIDAS phone photos only (no MIDAS dermoscopy); research-licensed data used now, a commercial-safe (CC-0/CC-BY) variant trained at stage C; batch 1 runs before S1 signs off (she can veto the recipe); batch 1 = reference seed 2, v4, v4 + diagnosis-balanced sampling (count^-0.5), v4 7-class head, 3 seeds each |
| 2026-10-21 | Second LLM judge | S2 | **Done:** Gemma-3-27B rerun (compact JSON via xgrammar) parses 100%; clean specificity 0.81 (Llama 0.79); judges agree on 90% of claims but kappa 0.35, so judge direction labels stay unused (H3) |
| 2026-10-21 | Classifier sprint stage B batch 1 (Narval 6691970–6691983) | S2 (prereg rule) | **Nothing adopted.** 3-seed mean S: reference 0.808; v4 (+ MIDAS phone photos) 0.799 (MIDAS dev_val AUROC 0.695 vs 0.714; seed SD 0.017 vs 0.004); v4 + diagnosis-balanced sampling 0.780; v4 7-class head 0.815 (+0.006) but internal-val specificity −0.026 and melanoma recall −0.10, so it fails the no-worse check. Fine-tuning on 1k phone photos does not reproduce the gain seen with frozen-feature probes; batch 2 targets representation (DINOv2 distillation, hygiene) and the phone-photo lesion localizer |
| 2026-10-21 | Interpretability prototypes on the interim model (I2/I3/I5/I7, selection data) | S2 | **Evidence:** the ResNet's embedding barely separates phone photos from close-ups (Mahalanobis AUROC 0.59) while DINOv2/MONET/PanDerm do (0.90–0.95), but no space flags which shifted images are wrong (≈ 0.5): a scope warning, not a deferral filter. Similar-case retrieval: 84% same-label neighbours internally, 62% on MIDAS (gate it by scope). The ResNet encodes pigment, erythema, vessels, keratosis look and texture (probe R² 0.67–0.83) but not shape moments (≈ 0). A linear model on named concepts matches the ResNet's call 87.1% (tree 84.8%) and explains about half its logit. **Proposed** for the freeze re-run (Michael to decide): cite only features the ResNet encodes, report the named-concept share, add a DINOv2 scope flag, compare a linear concept surrogate with the tree |
| 2026-10-21 | Explanation features after the net-benefit review (I2/I3/I5/I7, interim model) | Michael | **Decided:** (1) features the ResNet does not encode (all shape-asymmetry and border-geometry features, Hu moments, some colour fractions; linear and nonlinear probes R² < 0.3) are **labelled**, not dropped: 98% of current explanations cite one; (2) DINOv2-space **scope flag** + warning (MIDAS wrong predictions with a caution 58% → 78%; fires on 5% internally); (3) **named-concept share** sentence only when named features explain little (22% of cases; marks surrogate disagreement, not errors); (4) **similar-case panel not in the product**: in the ResNet's space it reinforces 80% of wrong predictions; a DINOv2-space version (contradicts 43% of errors, 15% of correct calls) is an optional clinician-study arm. Cost: +20 words per explanation; verifier pass rate unchanged (100%) |
| 2026-10-22 | Classifier sprint stage B batch 2 (Narval 6711643–6711671) | S2 (prereg rule) | **Nothing adopted.** 3-seed mean S vs reference 0.808: hygiene (AdamW, label smoothing, EMA) 0.817 (+0.008, but specificity −0.036, melanoma recall −0.13); +384 px 0.812; +DINOv2 distillation 0.805; +distillation on v4 0.806; +phone/skin-tone augmentation 0.810; DINOv2-L fine-tune 0.810 (internal AUROC 0.92 but MIDAS 0.70, same as the ResNet). Seed SD (0.004–0.027) exceeds the gaps, melanoma recall rests on 55 validation images, and neither a stronger backbone nor distillation lifts phone-photo AUROC above ~0.72: the limit is phone-domain data, not the recipe. Next: seed ensembles and weight soups (one ResNet50) |
| 2026-10-22 | Stage C: combine instead of more one-change arms | Michael | **Decided:** seed ensembles gain +0.011 to +0.020 selection score in all five groups (all-15 ensemble 0.835) while weight soups give nothing, so the product is one ResNet50 **distilled from a 12-model ensemble**. Batch 3: the student on split v4 scores 0.824 (+0.015, seed SD 0.004; internal AUROC 0.912, MIDAS 0.735), about 60% of the ensemble gain in one model, but fails the melanoma check (−0.115) like every new-trainer arm |
| 2026-10-22 | Melanoma | Michael | **Decided:** offer two modes (standard 90% sensitivity; melanoma-sensitive with a melanoma head: malignant if p ≥ t OR p(melanoma) ≥ t_m, t_m chosen on internal val for melanoma recall ≥ 0.65; on 55 validation melanomas a 7-class version lifted recall 0.43 → 0.65 at −13 points specificity). Add melanoma-rich clinical data to train only: Derm7pt clinical photos (252 melanomas) and the Asan/SNU sets (108; resolution shortcut equalized), research licences; ISIC clinical overview stays locked. Train melanomas 345 → 798 (split v6 = v4 + these). Batch 4: melanoma head, melanoma data, both, on the distilled student |
| 2026-10-07 | Generator LLM choice | S2 | **Provisional:** Mistral-7B-Instruct-v0.3 + Qwen2.5-7B-Instruct (Apache-2.0); Llama optional pending M-8 |

---

## 9. Reference map (where things came from)

- Objectives, methods, Gantt, milestones: `MedirAI Docs/2026 Grant Proposal.pdf` §2.1–2.4.
- Prior VLM/LLM description pipeline and prompt lessons: `Generative model architecture.docx`, `Report - end or internship_.docx`, `Work by Sasmi_ Explainability Improvements.docx`.
- Prior LLM-over-heatmap explanations: `XAI results.docx`.
- Dataset options for concept validation: `ABCD annotations.docx`, `Benchmarking Explainability Outputs.docx`.
- Diameter computation: `Computing Lesion Diameter.docx`.
- Prompt-testing protocol: `Reproducible Prompt Testing.docx`.
- Uncertainty and deferral baselines: `sprint_updates_zabboud.pptx-1.pdf`, `uncertaintyNet-main/`.
- Grad-CAM/SHAP/LIME code: `poc-nrc-main/medirai_dnn_explainability_engine.py`.
- Literature: Lundberg et al. 2020 (TreeSHAP); Zhang et al. 2023 (LLM narration of SHAP); Chanda et al. 2023 (dermatologist-like XAI and trust); Alfi et al. 2022 (handcrafted Hu/Haralick/color features + SHAP).
