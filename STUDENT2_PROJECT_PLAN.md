# Student 2 Project Plan: Feature Selection for LLM-Based Explanation Generation

**Project:** Explainable AI for Enhancing Deep Learning-Enabled Skin Cancer Detection (Mitacs Accelerate, IT 38322)
**Student 2:** Michael McKenna · **Student 1:** Shatha Albeetar · **Supervisor:** Prof. Adel Abusitta (Polytechnique Montréal) · **Partner:** MedirAI (J.-F. Djoufak)
**Duration:** 4 months, **Mon 2026-09-21 → Wed 2027-01-20**.
- Month 1 = W1–4 (Sep 21 – Oct 18)
- Month 2 = W5–8 (Oct 19 – Nov 15)
- Month 3 = W9–12 (Nov 16 – Dec 13)
- Month 4 = W13–17 (Dec 14 – Jan 20). It includes the Polytechnique holiday closure (~Dec 23 – Jan 5), so plan roughly three working weeks.
**Last updated:** 2026-10-02

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
| **M3 · W9–10** (Nov 16 – 29) | Prompt strategy comparison (deterministic); verifier; LLM-as-judge calibrated; case set for clinician review frozen | S1-6, S1-7, S1-9, A-1 cleared | Explanations for deferred cases → S1 |
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
- ☐ **T0.5** Kick off the ethics question with the supervisor (→ A-1). Clinician review in M3 is on the critical path.
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
  - ☑ Handoff Steps 1–4 prepared and verified locally (2026-09-25): `SHATHA_HANDOFF.md` (local), `splits/check_images.py`, `requirements*.txt`, and a git bundle. S1 still runs the tests, split check and image check on the VM.
  - S2-relevant P1 items: P1-15 (uniform logits export), P1-16 (Grad-CAM on the predicted class), P1-17 and P1-10 (mask quality and Dice).

### WS1: Shared feature layer (co-owned with S1) (M1 W3 → M2 W6)

- ☐ **T1.1** Agree on the feature schema with S1 (→ S1-2). One row per `isic_id` with columns `split`, labels, `mask_quality`, then the feature columns. Store as versioned parquet.
- ◐ **T1.2** Masks: get segmentation masks for the clinical close-ups (→ M-3). Apply the convex-hull cleanup from the NRC work (largest contour → hull). Compute a **mask-quality flag** (area ratio, border touching, confidence/uncertainty if Aloys's UNet provides it). Prior report: segmentation models "don't work with real (uncropped) images". *2026-10-02: v0 masks for all 7,983 images: ResNet18-UNet trained on ISIC 2018 (test Dice 0.894) + zero-shot SAM second opinion → ok/review/fail flag. Only ~51% ok/review: the dermoscopy-trained UNet over-segments small or faint clinical lesions. No convex hull (it erases border irregularity). Next: adapt to clinical photos with UNet–SAM agreement pseudo-labels; M-3 weights still wanted.*
- ◐ **T1.3** Implement **concept-aligned** feature families. Start from `extract_features.py` and fix it where needed: *2026-10-02: v0.1 = 44 features (A/B/C/D, texture, shape) at a common 512-px working size. Frame-relative D features let the features identify the MSKCC source (benign-only AUROC 0.90); v0.2 removes them.*
  - **A (asymmetry):** mask flip overlap about both principal axes, not just horizontal; color asymmetry across halves.
  - **B (border):** circularity, fractal dimension, convexity defects, border gradient sharpness (abrupt vs. fading edge).
  - **C (color):** count of clinically named colors (light/dark brown, black, blue-gray, red, white) in a perceptual space (CIELAB) instead of CSS3 nearest-name; color variance/entropy.
  - **D:** pixel Feret diameter, with the no-mm-calibration caveat.
  - **Texture/surface:** GLCM Haralick and LBP.
  - **Shape:** Hu moments, eccentricity, solidity.
  - Optional: PCA components of ResNet50 penultimate embeddings (→ S1-6).
- ☑ **T1.4** Unit tests on synthetic shapes and masks (a known-symmetric disc, a known number of colors) so every feature behaves as named. *2026-10-02: `tests/test_features.py` (symmetric disc, rotated ellipse, star border, two-color lesion, rough surface, resolution invariance).*
- ◐ **T1.5** Run on the full dataset on GCP → **feature table v1** plus a **feature dictionary** (name, concept group, units, direction meaning, valid range, extraction version). *(D1)* *2026-10-02: feature table v0.1 + dictionary + manifest built locally (329 failed-mask rows kept with an error flag); v1 after S1 review.*

### WS2: One-vs-all tree models, TreeSHAP, top-3 selection (M1 W3 → M2 W8)

- ☑ **T2.1** Define the one-vs-all classes. Use diagnosis classes with enough samples (e.g., melanoma, BCC, SCC, nevus, seborrheic keratosis, other). Decide how to handle "indeterminate" (exclude from training, keep for analysis). Log the decision in §8. *2026-10-01: BCC, Nevus, SCC (invasive + NOS + in situ), SK, Melanoma (invasive + in situ + NOS + metastasis), Other (keratoacanthoma stays in Other).*
- ☑ **T2.2** Train XGBoost/LightGBM one-vs-all models on the S1 split. Use class weights, tune on val only, and report per-class AUROC/F1. *2026-10-02 (v0): XGBoost, val-tuned; one-vs-all test AUROC 0.77 (SCC) to 0.87 (nevus, melanoma); macro-F1 0.44.*
- ☑ **T2.3** Compute TreeSHAP (`shap.TreeExplainer`, exact). Choose and document interventional vs. tree-path-dependent. Explain in log-odds space. *2026-10-02: tree_path_dependent, log-odds of malignant, signed toward ResNet50's predicted class.*
- ☑ **T2.4** **Concept grouping:** sum SHAP within each concept group (A, B, C, D, texture, shape, embedding) so correlated features don't fill all three slots. Rank concepts for the **predicted class**. Keep the top-3 with sign and magnitude bucket (strong/moderate/weak). *2026-10-02: grouped top-3 with strength tertiles fixed on val; most-cited groups color 29%, border 24%, size-in-frame 23%.*
- ◐ **T2.5** **Surrogate fidelity (critical).** Measure agreement between the tree prediction and the ResNet50 prediction (→ S1-3). Compare two variants: tree trained on **ground truth** vs. tree trained on **ResNet50's predicted labels** (a true surrogate of ResNet50). Pick based on fidelity (C1), and flag low-agreement cases in the explanation. *2026-10-02 (provisional): surrogate trained on ResNet50 labels = 84.9% agreement (benign 77%, malignant 91%) vs 82.0% for the ground-truth tree, so the surrogate is used; just under C1. 70% agreement on ResNet50-wrong cases.*
- ☑ **T2.6** Stability: check that the top-3 is stable across seeds and bootstrap resamples (Jaccard of top-3 sets). Report it. *2026-10-02: mean top-3 Jaccard 0.76 across 5 bootstrap refits.*
- ☐ **T2.7** Cross-check with S1's binary tree model and its TreeSHAP (→ S1-5). Where the malignant-vs-benign drivers disagree with the one-vs-all drivers, document why. *(D2)*

### WS3: Feature→dermatological-concept mapping (M1 W4 → M2 W8)

- ☐ **T3.1** Draft a **concept map and phrase bank**:
  - one entry per concept group, with graded phrases (e.g., asymmetry: "symmetric / asymmetric in one axis / asymmetric in two axes"; border: "regular / mildly irregular / markedly irregular"; color: "one color / two colors / multicolored, including blue-gray");
  - a definition of each term;
  - its ABCD or clinical relevance.
- ☐ **T3.2** **Calibrate the thresholds with data, not by hand:**
  - fit cut-points against labeled datasets: ENHANCE A/B/C scores (ISIC2017 + PH2), PH2 expert asymmetry and colors, Derm7pt criteria, and SLICE-3D metadata (border jaggedness, color variation, `clin_size_long_diam_mm`);
  - where no labels exist, use percentiles of the benign reference distribution.
- ☐ **T3.3** **Validate** each concept (C2): agreement between our computed concept grade and the expert labels. Drop or flag concepts that don't reach the target. Note the domain shift (these sets are dermoscopic; our main set is clinical close-ups).
- ☐ **T3.4** Embedding features, only if used: build an intermediate mapping from important embedding dimensions to known patterns, for example:
  - probing classifiers trained on Derm7pt criteria, or
  - concept activation vectors (TCAV-style), or
  - WhyLesionCLIP text-concept similarity.
  If none of these is reliable, label these features "learned visual pattern (not clinically mapped)" instead of inventing a concept.
- ☐ **T3.5** Send the concept map and phrase bank for clinical review (→ M-6). The prior texture definitions were never checked by a clinician. *(D3)*

### WS4: LLM setup and input structuring (M1 W3 → M2 W6)

- ☐ **T4.1** Shortlist models: Llama 3.x 8B Instruct and Mistral 7B Instruct (grant), plus optionally a biomedical-tuned variant. Check the licenses for commercial use (→ M-8). Serve locally on GCP (vLLM or HF transformers) so no data leaves MedirAI infrastructure.
- ☐ **T4.2** Define the **input JSON schema**. It contains:
  - the prediction and probability;
  - the uncertainty/deferral status and reason (from S1-4, e.g., "models disagreed", "high entropy");
  - surrogate agreement;
  - the top-3 concepts, each with phrase, direction ("supports malignant" / "argues against"), strength, and `feature_id`;
  - the target audience.
  It must never contain images, metadata, or identifiers.
- ☐ **T4.3** Define the **output JSON schema**: `summary`, `reasons[] {text, feature_id, direction}`, `uncertainty_note`, `limitations`. Add constrained decoding or a JSON grammar if the serving stack supports it.
- ☑ **T4.4** Build a **template baseline (no LLM)** that turns the same JSON into fixed sentences. It serves as the lower bound and the fallback. *2026-10-02: `explain/template.py`; every reason cites a feature_id; uncertainty, surrogate disagreement and mixed evidence always stated.*

### WS5: Prompting strategies (M2 W7 → M3 W10)

- ☐ **T5.1** Implement strategies:
  - **(a)** zero-shot structured prompt;
  - **(b)** few-shot, with care, because examples were copied verbatim in prior work;
  - **(c)** "cite-your-feature" prompt requiring a `feature_id` per claim;
  - **(d)** audience variants (dermatologist vs. GP; confirm the target with M-5).
- ☐ **T5.2** Protocol, following "Reproducible Prompt Testing":
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
- ◐ **T5.4** Explicit tests on **wrong** and **deferred** predictions (→ S1-9) so explanations don't lend false confidence. *(D4, prompt library)* *2026-10-02: done for the template and Grad-CAM (LLM pending). Confidently wrong predictions get explanations indistinguishable from confidently correct ones (surrogate agrees 92% vs 92%; no caution 62% vs 63%); only the uncertainty flag separates them.*

### WS6: Automated validation (M3 W9 → M4 W14)

- ☐ **T6.1** **Verifier** (rule-based + NLI): grounding rate (C3), top-3 coverage, direction correctness (C4), invented-concept detection against the concept vocabulary, and the uncertainty-flag rule (C5).
- ☐ **T6.2** **Counterfactual sensitivity:** perturb or swap input concepts, then check that the explanation changes accordingly and unchanged parts stay stable.
- ☐ **T6.3** **Consistency:**
  - similar inputs (same top-3 concepts and directions) should produce semantically similar outputs;
  - outputs should be robust to prompt paraphrase;
  - the same input run repeatedly should give identical output.
- ☐ **T6.4** **LLM-as-judge:**
  - use a different model family from the generator;
  - rubric: accuracy vs. input, coherence, clarity, clinical tone;
  - **calibrate the judge against human ratings** on ~50 samples (Spearman / κ) before trusting it at scale.
- ◐ **T6.5** **Stratified sampling:** correct vs. incorrect × confident vs. deferred × class. Report every metric per stratum, including C6 (confident language on wrong cases). *2026-10-02: strata implemented in `eval/wrong_prediction_study.py`; error-detection AUROC: model entropy 0.74, explanation support 0.65, Grad-CAM 0.54.*
- ☐ **T6.6** Compare: template baseline vs. each prompt strategy vs. each LLM. Report latency and GPU memory per explanation. *(D5)*

### WS7: Clinician review (prep M1–M2, run M3, analyze M4)

- ☐ **T7.1** Ethics determination and approval **before any clinician-facing session** (→ A-1).
- ◐ **T7.2** Study design (draft end of M2). Proposal: *2026-10-02: draft 40-case set (5 per correct/wrong × confident/uncertain × benign/malignant cell) with label-only, Grad-CAM and template materials and a separate answer key.*
  - **Case set:** ~40 cases stratified as in T6.5, frozen model version (→ S1-8).
  - **Sequential reveal per case:**
    1. The clinician's own impression, unaided.
    2. The model label and probability.
    3. Plus Grad-CAM *or* plus the NL explanation. Arm order is counterbalanced across clinicians.
  - **Ratings (5-point):** clinical relevance, trustworthiness, clarity. Also: "does this support or change your judgment?" and free text.
  - Include wrong-prediction cases to measure automation bias.
- ◐ **T7.3** Materials: rating form (Google Form or a simple web page), Grad-CAM images from S1's ResNet50 (→ S1-7), and a clinician instruction sheet. *2026-10-02: Grad-CAM arm built on the predicted class (P1-16); deletion AUC 0.656 vs 0.641 for random order, i.e. not faithful; 33% of CAM mass on the lesion (22% of the image).*
- ☐ **T7.4** Recruit and schedule clinicians (→ M-7). Target ≥ 3 dermatologists; hold sessions in M3 W11–12.
- ☐ **T7.5** Analysis: descriptive statistics per arm and stratum, inter-rater agreement, qualitative themes. Feed the findings back into the prompts and concept map. *(D6)*

### WS9: Dataset development (added 2026-10-02; MedirAI has no images yet, see M-4)

- ◐ **T9.1** Assess the current data. *2026-10-02: 7,983 labelled ISIC clinical close-ups (7,582 lesions). Malignant is 71% BCC; melanoma 522 images (348 train); few benign mimics (SK 575, LPLK 135, solar lentigo 63, dermatofibroma 52); only 1,378 benign images biopsy-confirmed; Fitzpatrick V–VI ≈ 5% of the 2,457 with a recorded type. Learning curve (25/50/100% of training groups, 2 runs each): test accuracy 0.805 → 0.830 → 0.840, AUROC excluding MSKCC 0.788 → 0.805 → 0.836, i.e. about +2 points per doubling, not yet flat. BCC/SCC are saturated (~0.97); nevus and SK keep improving; melanoma falls (0.63 → 0.54) as more non-melanoma data is added, so melanoma needs targeted data and class weighting, not just volume. Train ≈ val accuracy at every size, so the current recipe (lr 1e-5, 11 epochs) under-fits; tune it before concluding how much data is needed.*
- ☐ **T9.2** Check candidate public sources for licence, label provenance and overlap: MILK10k and PAD-UFES-20 (likely overlap with our MILK/UFES images), Derm7pt (melanoma-rich; also WS3), DDI (biopsy-proven, balanced skin tones; fairness test set), MED-NODE, Dermofit (paid; has lesion masks), Fitzpatrick17k and SLICE-3D (secondary). Dermoscopic sets only for pretraining.
- ☐ **T9.3** De-duplicate every new source against the current data by lesion and patient (and near-duplicate image hashing) before use; new sources go into a new split version (`isic_clinical_v2`), never by re-splitting v1.
- ☐ **T9.4** Datasheet for the assembled dataset: sources, licences, label provenance (histopathology vs clinical), class and skin-type balance, known confounds.

### WS8: Integration, documentation, reporting (M3 W11 → M4 W16)

- ☐ **T8.1** Package it as a module: `explain(isic_id | features, prediction, uncertainty) → {top_concepts, text, verification}`. Agree the integration target with MedirAI (→ M-10), e.g., a new endpoint in the payload-generation FastAPI.
- ☐ **T8.2** Integrated prototype with S1's refined ResNet50 (★ End M3).
- ☐ **T8.3** Re-run the full pipeline and evaluation on S1's final model (→ S1-8).
- ☐ **T8.4** Documentation: README, feature dictionary, concept map, prompt library, eval protocol, known limitations (no mm diameter, no "E", clinical-vs-dermoscopic domain shift, surrogate-fidelity caveat).
- ☐ **T8.5** Privacy/data-use summary contribution (*D8*), joint report with S1, Mitacs final report, exit survey.

---

## 6. Dependencies on Student 1, MedirAI, and the supervisor/admin

### 6.1 From Student 1 (Shatha)

| ID | What I need | Needed by | Blocks | Status |
|---|---|---|---|---|
| S1-1 | Final train/val/test split file for ISIC clinical close-ups (70/10/20, **grouped by lesion/patient**, keyed by `isic_id`, all diagnosis levels). Produced by CODEBASE_TODO P0-1/P0-2 (co-owned) | M1 W2 (Oct 4) | T0.6, T0.8, T1.5, T2.2 | ☑ `splits/isic_clinical_v1.csv` (built by S2, 2026-09-25). S1 to confirm it matches her image folder |
| S1-2 | Co-design of the feature schema and extraction code; agreement on mask source and quality rules | M1 W3–4 | WS1 | ☐ |
| S1-3 | ResNet50 checkpoint **retrained on `isic_clinical_v1` after the P0 fixes**, + per-image predictions/probabilities on val/test (CSV by `isic_id`). A preliminary version is fine at first; final at the End-M2 milestone | M2 W5 (prelim), W8 (final) | T2.5, T4.2 | ☐ |
| S1-4 | Per-image uncertainty outputs: entropy-of-expected, expected entropy, variational variance, multi-model agreement/deferral flag, and which models disagreed | M2 W8 | T4.2, T6.5 | ☐ |
| S1-5 | Binary tree model + its TreeSHAP values (for cross-checking against one-vs-all) | M2 W8 | T2.7 | ☐ |
| S1-6 | ResNet50 penultimate-layer embeddings on all splits, only if embeddings are used as features | M2 W6 | T1.3 (opt.), T3.4 | ☐ |
| S1-7 | Grad-CAM heatmaps from the ResNet50 for the frozen clinician-review case set | M3 W10 | T7.3 | ☐ |
| S1-8 | Refined ResNet50 (versioned) + re-exported predictions/uncertainty; one version frozen for the clinician study | M3 W10 (freeze), M4 W13 (final) | T7.2, T8.2, T8.3 | ☐ |
| S1-9 | Lists of misclassified and uncertain/deferred cases + S1's attribution findings on them | M3 W9 | T5.4, T6.5 | ☐ |

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
| M-5 | Explanation spec: target reader (dermatologist vs. GP; prior prompts targeted GPs), length, language (EN/FR?), where it appears in the product | M1 W2 | T4.2, T5.1 | ☐ |
| M-6 | Clinical reviewer (in-house advisor or contact) to sanity-check the concept map, phrase bank, and definitions | M2 W7 | T3.5 | ☐ |
| M-7 | Recruit ≥ 3 dermatologists for the review; confirm availability for M3 W11–12 and any compensation or logistics | Confirmed by M2 W8 | T7.4 | ☐ |
| M-8 | LLM licensing/deployment constraints: Llama 3 Community License vs. Mistral (Apache-2.0); on-prem only? | M1 W2 | T4.1 | ☐ |
| M-9 | Sprint check-in cadence (weekly or biweekly), shared repos/Drive, who attends | M1 W1 | T0.4 | ☐ |
| M-10 | Integration target and interface for the module (e.g., payload API endpoint), plus handoff format | M3 W11 | T8.1 | ☐ |

### 6.3 From the academic supervisor / administration

| ID | What is needed | Needed by | Blocks | Status |
|---|---|---|---|---|
| A-1 | Ask Polytechnique's research ethics office (CER) whether certification is required for (a) secondary use of ISIC and possibly MedirAI images and (b) the clinician review. Submit if required. **The grant §3.2 declarations currently answer "No" to human participants and secondary health data, while §2.2 plans a clinician review. Reconcile this.** | Started M1 W1; cleared by M3 W10 | WS7 (critical path) | ☐ |
| A-2 | Mitacs Part 2 completeness. Confirm approval status (Mitacs requires approval before the project begins). Open gaps in the PDF: Intern 1 details (§4.3.1), Section 5 suggested reviewers (empty), ORS signature (§7.5), Faculty field (§4.2.1), supervisor name typo "Adel Aby Sitta" (§7.2), duplicated Month 4 Gantt row for Student 2 | ASAP | Funding, timeline | ☐ |
| A-3 | IP: extension of the existing research agreement (§3.9) | Before handoff (M4) | T8.1 | ☐ |
| A-4 | Agree on the success criteria (§1) and the clinician-study design (T7.2) | M1 W2 / M2 W8 | T0.4, T7.2 | ☐ |

---

## 7. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tree model has low fidelity to ResNet50, so the explanation describes a different model | Explanations are unfaithful | T2.5: surrogate-on-ResNet50-labels variant, report C1, flag disagreement cases in the text |
| Handcrafted features depend on segmentation, and masks fail on clinical close-ups | Wrong concepts | Mask-quality flag (T1.2); abstain from affected concepts; say "segmentation uncertain" |
| Concept thresholds don't match clinical judgment (prior color/texture rules ~27–30%) | Misleading phrases | Data-calibrated thresholds + validation (T3.2–3.3); drop weak concepts; clinical review (M-6) |
| Dermoscopic validation sets vs. clinical close-up target | Validation doesn't transfer | Report the domain shift; validate on SLICE-3D/PAD-UFES clinical metadata where possible |
| LLM invents features or overstates confidence (confirmation bias seen in prior XAI work) | Unsafe output | Structured output + verifier (T6.1); tests on wrong/deferred cases (T5.4); template fallback |
| Ethics approval or clinician recruitment slips | D6 late | Start A-1 in W1; recruit via M-7 by W8; fallback: pilot with fewer clinicians + stronger automated eval |
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
| — | Use ResNet50 embeddings as features? | S2+S1 | Open: only if the intermediate concept mapping is reliable (T3.4) |
| — | Target reader and language of explanations | M | Open (M-5) |
| 2026-10-02 | Proprietary MedirAI data in scope? | M | **Answered:** MedirAI has no images yet. We assemble the dataset ourselves from public sources (WS9) |
| — | Ethics certification required? | A | Open (A-1) |
| — | Generator LLM choice (Llama vs. Mistral) | S2+M | Open (T4.1, M-8) |

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
