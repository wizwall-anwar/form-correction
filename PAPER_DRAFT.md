# A Phone-Only, Abstaining Form-Feedback Prototype for the Squat: Clinician-Score Benchmark, Synthetic Multi-View Degradation, and a Grounded VLM Critique

**Author:** Anwar Ayoon (MIS 599 Creative Component, Ivy College of Business, Iowa State University)
**Advisor:** Anthony Townsend
**Draft date:** 2026-11-25
**Status:** Working manuscript for advisor review. All numerical results come from `results_bundle (4)/` (second clean Colab run, KIMORE Ex5, 75 rows, Gemini backend live). Two fields — per-flag Spearman ρ and detector-vs-clinician κ — have their helper code wired into `run_evaluation` and the notebook but are not yet in the bundle; they are explicitly marked `[NEXT-RUN]` and will populate on the next detector-cell execution (no Gemini required).

---

## Abstract

We build a phone-only squat form-feedback prototype that (i) predicts the KIMORE Exercise 5 clinician score from three deterministic kinematic flags (knee depth, trunk lean, left–right symmetry) under subject-level splits, (ii) quantifies how both flag measurements and score prediction degrade when the 3D skeleton is rotated off-plane (synthetic views, 0°–60°), and (iii) delivers a 3-bullet critique that is verified against detector measurements and refuses when confidence is low. VLM-vs-clinician agreement is framed strictly as *model-vs-clinician reliability*, never as validity. The headline finding is a **negative result**: on 75 subjects with leakage-safe splits, the ridge-on-flags baseline inverts the clinician score — Spearman ρ = −0.52 (95% CI [−0.77, −0.16], n_test = 23), MAE 0.075 vs mean-baseline MAE 0.069. The CI never crosses zero, and the inversion is stable across all five view angles (ρ ∈ [−0.54, −0.47]), so the failure is in how the flags combine, not in the view-rotation layer. Multi-view flag error itself is small (1.8°–2.6° mean across 15°–60°). Calibrated abstention under this degenerate 3-class projection gives ECE = 0.86 and a selective risk that is *highest* at the most-confident 13% of predictions — the classifier is most wrong where it is most confident. Gemini agreement with clinicians is near zero (κ = −0.14 at 0° on n = 55, +0.14 at 45° on n = 35), consistent with noise at these sample sizes. The grounded verifier rejects planted fabrications and banned terms as unit-tested; hallucination-rate figures remain a verifier control, not a measured VLM baseline. The contribution is the evaluation pipeline, the pre-registered honest reporting of a negative result, and a reusable set of claim-discipline rules.

---

## 1. Problem and Motivation

### 1.1 What this prototype is, and what it is not

Most movement-quality systems that reach published accuracy on squat form rely on specialized hardware (IMUs, depth sensors, multi-camera rigs). The question motivating this work is narrower and more deployable: what can a phone-only pipeline measure honestly, and where must it refuse rather than guess?

The prototype:

1. Computes three named, deterministic joint-angle flags on a squat sequence (knee depth, trunk lean, left–right symmetry).
2. Trains a small calibrated detector against KIMORE Exercise 5 clinician scores under a strict subject-level split.
3. Rotates the 3D skeleton through synthetic off-plane views and measures how flag error and prediction quality degrade.
4. Delivers a 3-bullet PT-style critique whose every number and deviation name must be verified against the detector output, or the sentence is replaced by a template; and refuses entirely when `abstain_flag=True`.

### 1.2 Discipline the design enforces

Three claims are banned anywhere in the pipeline, the demo, and this paper:

- **No injury language.** Flags are measurements, not verdicts about a person's body.
- **No VLM-as-ground-truth.** VLM vs clinician agreement is reported as *model-vs-clinician reliability*, never as validity.
- **No synthetic-as-real.** Rotated skeletons are SYNTHETIC views, with no occlusion, pose-estimator error, or clothing effects; every table caption that uses them says so.

### 1.3 Evidence anchors from the literature

(To expand with citations in the final draft; recheck before camera-ready.)

- MediaPipe knee flexion in squats shows mean bias ≈ −13.9° (SD 15.1), with limits of agreement [−43.5°, +15.5°] (Panni et al.). This motivates abstention rather than confident coaching on monocular pose.
- Subtle-error F-scores in Fitness-AQA sit near 53 (knees inward) and 46 (overhead-press elbow) — far from product-grade.
- FormCoach reports VLM hallucination rates of ≈ 74–97% across models on exercise critique tasks (arXiv:2508.07501), motivating the grounded verifier.
- Rehabilitation-AQA state of the art reports Spearman ρ ≈ 0.80–0.90 on KIMORE and UI-PRMD — the target the flag-based baseline is compared against as a skyline. This paper does not clear that skyline; it cleanly misses it in a measurable direction.

---

## 2. Architecture

The pipeline is five modules plus a constrained demo, each small enough to read and debug end-to-end.

### 2.1 `m00_data` — ingest with bounded fallbacks

Loads KIMORE Exercise 5 from a verified local pickle; falls back to UI-PRMD; synthetic last. A 30 s timeout per step and a single 15 s availability check for the gated Princeton 7-mistake labels prevent any network stall. The data source actually used is written to `results/manifest.json` and stamped on every downstream artifact. `resolve_data_dir()` is a bounded `os.walk` (depth 5) so Colab's extra `data/` wrapper and Kaggle's mount prefix do not silently fall through to the synthetic fallback.

### 2.2 `m01_features` — deterministic kinematics and synthetic views

`rotate_view(joints, deg)` rotates about the vertical axis; `add_jitter` adds seeded Gaussian noise; `project_2d` is orthographic. `kinematic_flags` returns three measurements:

- **Knee depth** — maximum knee flexion across the rep (deepest point).
- **Trunk lean** — maximum trunk angle from vertical.
- **Symmetry** — maximum absolute left–right knee-angle difference.

Thresholds live in `src/config.py`, tuned on training subjects only. Flags are measurements; `*_pass` booleans are reported but never interpreted as correctness verdicts for a person.

### 2.3 `m02_detector` — subject-safe baseline, calibration, abstention, per-flag diagnostic

- **Splits** are grouped by `subject_id`; `run_evaluation` asserts four-way disjointness across fit, validation, test, and the parent train group. A deliberately leaky clip-split is kept only for a leakage comparison row.
- **Regression (primary).** Closed-form ridge on the three flag measurements predicts the raw clinician score. Reported as Spearman ρ and MAE on held-out subjects, with subject-bootstrap 95% CIs, and a train-mean MAE baseline so the reader can see whether the flags beat *predict the average*.
- **Classification (secondary).** Multinomial logistic regression over tertile bins; used only for calibration (temperature scaling on a held-out validation subject group) and abstention (threshold on calibrated confidence).
- **Per-flag diagnostic (new).** `per_flag_correlations(samples)` returns Spearman ρ of each flag vs the clinician score on held-out subjects; `run_evaluation` emits it under `per_flag_subject_split`, and `build_table1` surfaces it. This decomposes the headline inversion by flag so §5 can name the suspected mechanism rather than guess it.
- **Risk-coverage** feeds Figure 1 (`fig1_risk_coverage.png`).

### 2.4 `m03_agreement` — VLM vs clinician as reliability

Keyframes (start, bottom, end) are rendered as stick-figure PNG triptychs from the rotated, projected skeleton. The VLM is prompted for a single-digit bin (0/1/2). Cohen's κ (unweighted and quadratic-weighted) is reported with subject-bootstrap CIs. Reporting language is locked to *model-vs-clinician reliability*. Failed VLM calls return `STUB — no result` and are **excluded** from κ; they are never replaced by a placeholder bin.

Resilience (in response to transient `503 UNAVAILABLE` from Gemini): the caller walks a `GEMINI_MODEL_FALLBACKS` list, honors `Retry-After` headers, uses exponential backoff with jitter bounded by `VLM_RETRY_MAX_ATTEMPTS`, `VLM_RETRY_BASE_DELAY_S`, and `VLM_RETRY_MAX_DELAY_S`. Successful calls are cached on disk under `results/vlm_cache/` by `(seq_id, angle, seed)`, so a resumed notebook run does not re-spend quota.

Comparison line (new helper, wired for next run): `detector_vs_clinician_kappa(detector_bins, clinician_bins)` rates the flag-based classifier against the same clinician bins and surfaces in Table 4 under `detector_vs_clinician_kappa`. This gives a ground baseline for the VLM κ rather than comparing against zero.

Fleiss' κ for VLM self-consistency across repeated runs is listed as *future work*. No repeated-run data was collected on the Gemini backend in the current run; the earlier framing of a measured Fleiss' κ is demoted to §6.

### 2.5 `m04_report` — four tables, one figure, one zip

`build_table1` through `build_table4`, `fig1_risk_coverage.png`, `manifest.json`, and `results_bundle.zip`. Any artifact built from synthetic data is stamped `SYNTHETIC — not evidence`. Captions on Table 2 and Table 4 state that the views are synthetic rotated views, not real camera views. `build_table1` now accepts an optional `per_flag_subject_split` kwarg; `build_table4` accepts an optional `detector_vs_clinician` kwarg. Both are backward-compatible: omitted kwargs leave the field absent.

### 2.6 `app/biocoach_demo` — the constrained critique

Input is one detector record (`flags`, `abstain_flag`, `confidence`, `view_deg`).

- If `abstain_flag=True`: exactly one refusal sentence, no critique.
- Otherwise: exactly three bullets from a fixed template.
- `verify_claims` rejects any sentence containing a number that does not match a detector value within ±0.05°, and any sentence containing a banned term (injury, hurt, damage, risk, diagnose, treat, …). The sentence is replaced by the template.
- `log_hallucination_comparison` records unsupported-number counts for free-form text vs grounded bullets, feeding Table 4.

### 2.7 `scripts/make_slide_exhibits.py` — static presentation exhibits

An offline plain-Python script (same pattern as DECISIONS §5, 2026-11-25) that reads the KIMORE pickle (falls back to the synthetic generator), renders start/bottom/end stick-figure keyframes at 0° and 45°, and composes a side-by-side `fig_biocoach_safety_exhibit.png` showing the 45° triptych next to the grounded BioCoach bullets and a rejected free-form fabrication (`999.0°`). The exhibit uses the actual detector numbers from `halluc_log.jsonl` record 1 (knee 106.72°, trunk 26.85°, symmetry 19.59°). This replaces a live demo during the talk — the committee sees the pipeline's visual output and the verifier's rejection behavior side-by-side without any network dependency.

---

## 3. Dataset and Leakage-Free Evaluation

### 3.1 KIMORE Exercise 5 (primary)

The owner's licensed copy is a single pickle, `kimore_exercise_dataset.pkl`, whose `ex5` key is a pandas DataFrame of **75 rows**, one per sequence, with 25 lowercase joint-name columns and a `cTS` clinician-score column. Each joint cell is a `(T, 7)` array whose columns 0:3 are a unit quaternion (verified `‖q‖ = 1.0`) and columns 4:7 are XYZ position; only the position is used. This XYZ-column choice is independently validated by physiological knee geometry (§4.1).

**Score distribution (local plain-Python read of the pickle, not a pipeline run).** `cTS` is min 0.50, Q1 0.83, median 0.89, Q3 0.96, max 1.00, std 0.10, with 37 distinct values across the 75 sequences. The distribution is identical across all five exercises in the pickle, consistent with (but not proof of) one row per subject.

**Honest limitation — score clustering.** 75% of scores sit at 0.83 or higher; the top quartile is compressed in a 0.04-wide band. An earlier tertile-classification result (accuracy 0.96) was invalidated by a bug in `_score_bin` that used the quantile *probabilities* (1/3, 2/3) as absolute score cutoffs, producing a 72/3/0 split. Fixed: true tertile edges (≈ 0.86 and ≈ 0.94) give balanced 24/24/27 bins, and tertile edges are now computed on **training subjects only** inside `m02_detector`. Datasets with fewer than three distinct scores (e.g. UI-PRMD's binary labels) fall back to rank bins.

**Pivot to regression.** Because scores are compressed near the top of the scale, a 3-way class label discards signal. The primary detector is ridge regression on the raw score, with tertile classification kept only for calibration and abstention (Table 3). We report Spearman ρ, MAE on held-out subjects, subject-bootstrap 95% CIs, and the train-mean MAE as a constant baseline.

### 3.2 UI-PRMD (fallback) — parent-relative kinematics caveat

UI-PRMD ships binary correct/incorrect labels, and its "incorrect" repetitions were deliberately performed by healthy subjects — not natural errors. The loader uses the paper's Table 2 order (Kinect positions, 22 joints × 3 = 66 columns per row); segment names such as *upper leg* / *lower leg* / *foot* are anatomical approximations for our standardized ten-joint slots, and the loader emits a warning whenever UI-PRMD samples are actually produced. Any clinician-vs-model agreement claim is suppressed or marked unavailable when UI-PRMD is the backing dataset. The loader accepts both the Zenodo archive folder names (`Segmented Movements`, `Incorrect Segmented Movements`) and the earlier `Correct`/`Incorrect` mirror layout, requires exactly 66 finite numeric values per row, and keeps all repetitions of a subject under one split key.

### 3.3 Princeton 7-mistake labels

One 15 s availability check; if gated or unreachable, the pipeline proceeds without them. Currently returns `None`.

### 3.4 Leakage guard

Every evaluation path asserts `len(set(train_subj) & set(test_subj)) == 0`. The test is in `tests/test_leakage.py` and runs in CI. `run_evaluation` additionally asserts disjointness across all three pairs of the fit / validation / test subject groups, after an earlier notebook cell was found calibrating on the test subjects.

**Honest limitation — clip-vs-subject leakage comparison is not informative on this data.** With 75 subjects and 75 sequences (`max_sequences_per_subject == 1`), the clip-split cannot leak by construction: every subject contributes one sequence, so a random sequence split is also subject-safe. Table 1 carries the `max_sequences_per_subject: 1` field and a note stating this explicitly. On the current bundle the clip-split ρ (−0.30) and subject-split ρ (−0.52) differ, but the difference is not a leakage signal — it is two separate random partitions of the same 75 rows (the subject split shuffles subjects; the clip split shuffles sequences), landing on different 23-row held-out subsets of the same 75 subjects. The variance between the two ρ values reflects random-subset variance on a small dataset, not leakage. We do not report this as *"leakage did not matter"*; it was untestable on KIMORE Ex5 in its current form.

---

## 4. Verified Results

Numbers in this section come from `results_bundle (4)/` (second clean Colab run, 2026-11-25, Gemini backend live). Fields marked `[NEXT-RUN]` have their helper code wired into `m02_detector.run_evaluation` and the notebook (Patches A and B, 2026-11-25) and will populate on the next detector-cell execution — no Gemini call is required for either.

### 4.1 Physiological flag extraction on KIMORE Ex5 [VERIFIED]

An offline inspection of the real pickle (plain-Python read, not a pipeline run, per CLAUDE.md §3) measured knee flexion on KIMORE ex5 subject 0 at the deepest point of the rep. Values peaked at **≈ 99–111°** with left ≠ right — physiological for a deep squat and consistent with the published range. This also validates the XYZ-column choice in the pickle (columns 4:7 of each (T, 7) joint cell): a plausible XYZ permutation could not produce the physiological curve by accident, because the inter-vector angle is axis-order-invariant.

A minor known imprecision is logged, not fixed: `KINECT_V2_TO_STANDARD_JOINT_MAP` collapses `AnkleLeft` and `FootLeft` into one `ankle_left` slot (we have no foot slot), and in the parser the foot overwrites the ankle, shifting the knee-flexion peak from ≈ 108° to ≈ 99°. This is acceptable for a prototype; for finer knee accuracy, the parser should prefer the true `Ankle*` joint over the collapsed foot.

We also caught and corrected a knee-depth bug: `kinematic_flags` had taken the *minimum* knee flexion across the rep (the standing moment, ≈ 0°) rather than the maximum (deepest point), so `knee_depth_pass` could never be `True`. Switched to the maximum and renamed `knee_flexion_min_deg` → `knee_flexion_max_deg` across `m01`, `m02`, the demo, and the tests.

### 4.2 Primary finding — the ridge-on-flags baseline inverts the clinician score [VERIFIED, HEADLINE]

On held-out KIMORE Ex5 subjects (n_test = 23), the ridge regression on the three flag measurements predicts the clinician score **backwards**:

| Metric | Value | 95% CI (subject bootstrap) |
|---|---|---|
| Spearman ρ (held-out) | **−0.515** | **[−0.771, −0.156]** |
| MAE (held-out) | 0.0752 | [0.0535, 0.1000] |
| MAE mean baseline | 0.0690 | — |
| n_test | 23 | — |
| data_source | manual (KIMORE Ex5 pkl) | — |

**The 95% CI never crosses zero**, so this is a consistent inversion, not noise. The model also fails the "beat predict-the-average" test (MAE 0.075 > 0.069). Per DECISIONS §6 (pre-registered pivot criterion: *"Learned detector fails to beat the simple flag/classical baseline on held-out subjects: report the baseline as the finding"*), this is the finding.

**Suspected mechanism.** Clinicians reward *controlled* form; the flags reward *motion magnitude* (deeper squat + some trunk lean + some left–right asymmetry). A single linear mixture must then trade off against itself, and on KIMORE's compressed top-quartile score band the mixture that best explains variance inverts the sign.

**Per-flag diagnostic [NEXT-RUN].** `per_flag_subject_split` will carry {`knee_flexion_max_deg`: ρ, `trunk_lean_max_deg`: ρ, `symmetry_diff_deg`: ρ} on held-out subjects on the next notebook execution. A figure `slides_figures/fig_per_flag_vs_score.png` exists and is referenced from slide 7 for qualitative inspection; its origin (which subset of samples it was computed on) is not audited in this draft, so no quantitative ρ from it is cited here. The authoritative held-out per-flag ρ comes from the patch-A helper and is what the paper will cite on the next run.

### 4.3 Synthetic multi-view degradation (0° → 60°) [VERIFIED, SYNTHETIC VIEWS]

Table 2, reproduced from `results_bundle (4)/table2.json`:

| View (synthetic rotation) | Spearman ρ | MAE | Mean flag_error (deg) | is_synthetic_view |
|---|---|---|---|---|
| 0° | −0.536 | 0.0759 | 0.000 | false |
| 15° | −0.527 | 0.0751 | 2.106 | true |
| 30° | −0.471 | 0.0753 | 1.838 | true |
| 45° | −0.516 | 0.0756 | 2.569 | true |
| 60° | −0.521 | 0.0756 | 2.058 | true |

Caption: *"Detector performance and flag error for SYNTHETIC rotated views, not real camera views."*

Two readings:

1. **Angle stability is fine.** ρ stays in [−0.54, −0.47] across 0°–60°; MAE is flat at 0.075. The inversion in §4.2 is a *feature* problem, not a *rotation* problem — the detector is stably wrong.
2. **Flag error is small.** Mean flag error is 1.8°–2.6° at 15°–60°, with 0° serving as the reference (flag_error = 0 by construction). This is a legitimate standalone engineering finding: under the best-case geometric lower bound (no occlusion, no pose-estimator error, no clothing), the three deterministic flags survive synthetic yaw rotation to within 2.6° on average.

The figure `slides_figures/fig_multi_view_degradation.png` plots the per-angle ρ and MAE curves.

The degradation curve is the point: a pipeline that does not know it is looking at a bad view will keep coaching confidently against measurements that no longer reflect the real movement. Abstention (§4.4) is the design response.

### 4.4 Calibration, abstention, and grounded verifier [VERIFIED]

From `results_bundle (4)/table3.json`:

- **ECE = 0.861** on held-out subjects (3-class task, n_test = 23, test_class_counts = {0: 10, 1: 3, 2: 10}). The classifier is very poorly calibrated.
- **Risk-coverage is inverted.** Selective risk is **1.0** at the most-confident 13% of predictions (coverage = 0.043 / 0.087 / 0.130), drops to 0.75 at 17% coverage, and climbs back to 0.87 at full coverage. The classifier is **most wrong where it is most confident** — consistent with the regression inversion in §4.2.
- **Selective accuracy is below the 3-class random floor.** `selective_accuracy_at_coverage` returns {0.5: 0.083, 0.8: 0.105, 1.0: 0.130}. Random accuracy for a 3-class problem is ≈ 0.333, so even the top-confidence half of predictions is worse than guessing. This is the quantitative face of the §4.2 inversion.

Per DECISIONS §6 (pre-registered: *"If abstention gives no precision gain at matched coverage, report it as a negative result. Keep the abstain gate in the demo as a safety behavior, not a claimed contribution."*), the abstention gate is retained in `app/biocoach_demo.py` for honest refusal behavior, but it is **not** claimed as a performance contribution on this data.

**Anti-hallucination verifier [VERIFIED by unit test].** `verify_claims` and `_contains_banned_term` are unit-tested on two planted fabrications and both are rejected:

- *"Knee depth measured at 999.0 degrees."* → fabricated number, ≠ any detector value within ±0.05° → replaced by the template.
- *"Knee depth measured at 82.4 degrees; injury note."* → matches a detector value but contains the banned term `injury` → replaced by the template.

Both assertions are in `_smoke_test` and run in under 5 s offline.

**Refusal behavior.** `format_bullets` returns exactly one sentence — *"The view is unreliable, so no critique is given."* — when `abstain_flag=True`, and exactly three measurement sentences otherwise.

**Static safety exhibit.** `scripts/make_slide_exhibits.py` writes `slides_figures/fig_biocoach_safety_exhibit.png`: the 45° stick-figure triptych beside the grounded bullets (`Knee depth: maximum knee flexion measured at 106.72 degrees.` …) with a red-X free-form fabrication (`"The knee flexion is 999.0 degrees and trunk lean is 888.0 degrees."`) rejected by the verifier. Numbers in the exhibit come from `halluc_log.jsonl` record 1 — the real detector output on one KIMORE Ex5 sequence in the shipping bundle.

### 4.5 Gemini vs clinician agreement [VERIFIED, SMALL-SAMPLE]

From `results_bundle (4)/table4.json`, Gemini backend live (model walk `gemini-3.1-flash-lite` → `gemini-2.5-flash-lite`, retry + backoff + `Retry-After` + disk cache):

| View (synthetic rotation) | Cohen's κ | Quadratic-weighted κ | n_scored | n_stub |
|---|---|---|---|---|
| 0° | **−0.139** | −0.215 | 55 | 20 |
| 45° | **+0.139** | +0.221 | 35 | 40 |

Framed as **model-vs-clinician reliability**, not validity. Both κ values are near zero and the sign flip between 0° and 45° is plausibly noise at these sample sizes (n_scored = 55 vs 35; the 45° cell dropped 40/75 to 503s that even the retry/fallback chain could not convert). Gemini scored almost every sample as 0 or 2 and almost never 1 — the prompt or the stick-figure keyframe render appears to push to the extremes.

**Detector-vs-clinician κ [NEXT-RUN].** Helper `detector_vs_clinician_kappa` is wired into the notebook (reports on all 75 samples using dataset-level tertile bins — the paper notes the mismatch with the detector's train-only edges). It will appear in Table 4 as a comparison line on the next detector-cell run.

**Hallucination rates are a verifier control, not a measured VLM baseline.** `hallucination_rates` in Table 4 reads `{"free_form": 2.0, "grounded": 0.0}` with the explicit note: *"free_form is a PLANTED fabricated control string (999/888), not VLM output; this is a verifier check, not a measured hallucination rate."* The 999/888 string lives in notebook cell 14, not in `m03`. A real free-form-vs-grounded comparison requires actual free-form VLM text under the same prompt, which the current runner does not collect, and is listed as future work in §6.

---

## 5. Discussion

### 5.1 What we can and cannot claim

**Can claim.**

- The flag-based detector *inverts* the KIMORE Ex5 clinician score on held-out subjects, with a CI that does not cross zero (§4.2).
- The inversion is stable across synthetic view rotations (ρ ∈ [−0.54, −0.47] across 0°–60°), so the failure is in how the flags combine, not in the view-rotation layer (§4.3).
- Under synthetic yaw rotation, deterministic flag measurements survive with 1.8°–2.6° mean error at 15°–60° (§4.3).
- The classifier is miscalibrated (ECE = 0.86) and its selective risk is inverted on this task, so calibrated abstention gives no precision benefit; the gate is kept only for honest refusal behavior (§4.4).
- Gemini-vs-clinician κ is near zero at 0° and 45° (§4.5), framed strictly as model-vs-clinician reliability, with n_scored reported for each cell.
- The grounded verifier rejects planted fabricated numbers and banned terms as unit-tested (§4.4).

**Cannot claim.**

- That clinicians agree on correct form (KIMORE ships a combined score).
- That rotated skeletons behave like real off-plane camera footage (no occlusion, no pose-estimator error, no clothing).
- That results transfer to phone video (Kinect-to-phone untested).
- That any flag means injury risk or incorrect form for any person.
- That a VLM persona is a qualified evaluator.
- That UI-PRMD "incorrect" reps are natural errors.
- That "hallucination" is a claim about real-world truth; here it is solely *unsupported by detector-verified facts*.

### 5.2 Engineering pivots in service of honesty

- **Classification → regression** because scores are compressed near 1.0 and a 3-way label was discarding signal. Classification is retained only for calibration and abstention.
- **Hand-rolled numpy over scikit-learn** because `import sklearn` alone spent 5–7 s cold on the dev machine, blowing the hard 5 s smoke-test budget. The Colab/Kaggle path is free to use scikit-learn.
- **UI-PRMD Table 2 mapping over Kinect v2 25-joint assumption** after Firecrawl verification showed UI-PRMD is 22-joint Brekel order, not Kinect v2 25-joint. Logged in DECISIONS §10.
- **Three-way fit/val/test subject split** after an earlier notebook cell was found calibrating on test subjects.
- **`resolve_data_dir()` made recursive** because Colab's extra `data/` wrapper and Kaggle's mount prefix meant the old fixed-path checker silently fell through to synthetic.
- **`_score_bin` tertile-probabilities-as-cutoffs bug** fixed; the earlier 0.96 accuracy from the 72/3/0 bug run is void.
- **Max knee flexion over min** across the rep; `knee_depth_pass` would never have been True otherwise.
- **`selective_accuracy_at_coverage` replacing `precision_at_coverage`** — the old output was an empty dict.
- **m03 Gemini resilience** — model-list fallback, `Retry-After`, exponential backoff with jitter, on-disk cache.
- **Per-flag correlation diagnostic and detector-vs-clinician κ comparison row** (2026-11-25, patches A/B) — both wired into `run_evaluation` and the notebook so the mechanism of the §4.2 inversion is diagnosable without a second Gemini run.

### 5.3 Abandon / pivot criteria (pre-registered)

Pre-registered in DECISIONS §6 before the Colab run:

- *If the learned detector fails to beat the simple flag/classical baseline on held-out subjects, report the baseline as the finding.* **Triggered.** §4.2 is the paper's headline.
- *If abstention gives no precision gain at matched coverage, report the negative result and keep the abstain gate in the demo as a safety behavior.* **Triggered.** §4.4.
- *If grounded output does not lower the unsupported-claim rate versus free-form, drop that claim. Keep the app as a demo.* **Not yet triggered** — current hallucination numbers are a verifier control, not a measurement. Deferred to future work pending actual free-form VLM text collection.
- *If clip-split and subject-split numbers match, say so. Leakage did not matter on this data.* **Not evaluable.** `max_sequences_per_subject == 1`, so the clip split cannot leak by construction; Table 1 says so explicitly.

---

## 6. Limitations and Future Work

1. **Kinect-to-phone transfer is untested.** The whole pipeline uses Kinect-derived skeletons; a next step is to run BlazePose (or equivalent) on KIMORE RGB and re-measure both the §4.2 regression and the §4.3 degradation curve.
2. **Score field semantics are not fully documented.** KIMORE's `cTS` has no authoritative schema we could locate (DECISIONS §10); a licensed-copy audit of score direction and per-rater availability remains an open item.
3. **UI-PRMD joint-identity is unverified below the Table 2 slot level.** The parser maps segments to standardized landmarks anatomically, not by name equivalence.
4. **Clip-vs-subject leakage comparison is untestable on KIMORE Ex5 in its current form** (1 sequence per row). A dataset with multiple reps per subject would let us show whether the gap is real on this task.
5. **Hallucination rates are a verifier control, not a measured VLM baseline.** A real free-form-vs-grounded comparison needs actual VLM free-form text under the same prompt; the runner does not collect it yet.
6. **Fleiss' κ for VLM self-consistency is unmeasured on the Gemini backend** (no repeated-run data in the current bundle). Listed as future work rather than reported.
7. **Per-flag ρ and detector-vs-clinician κ** are helper-wired but not yet in the shipping bundle; both appear on the next detector-cell run and will replace the `[NEXT-RUN]` markers in Tables 1 and 4.
8. **Inversion mechanism needs a confirmation experiment.** §4.2 names a suspected mechanism ("flags reward magnitude, clinicians reward control"). A clean test is a non-linear model (e.g., GBM on the same three flags) or a feature that explicitly encodes smoothness / variance (joint velocities, hip-knee coupling, trunk-knee phase). This is the next experiment to run, and is also logged as future work in DECISIONS §5 (2026-11-25 second entry).
9. **Sample size on Gemini κ at 45° is small (n_scored = 35).** The sign flip between 0° and 45° should be interpreted as consistent with noise at these sample sizes, not as evidence of angle-dependent VLM behavior. Re-running cell 6 off-peak (so the ~65 cached-but-stubbed samples retry successfully) would tighten the 45° cell toward n = 55+.

---

## 7. Reproducibility

- **Code:** `github.com/wizwall-anwar/form-correction` (modules `m00`–`m04`, `app/biocoach_demo.py`, `kaggle_colab_runner.ipynb`, `scripts/make_slide_exhibits.py`).
- **Smoke tests (local, offline, < 5 s each):** `python -m src.m00_data --smoke-test` and the m01–m04 and biocoach counterparts.
- **Full test suite:** `PYTHONPATH=. pytest -v` — 74 tests, 65 pre-existing + 9 for the 2026-11-25 patches (per-flag correlations, detector-vs-clinician κ, Table 1 / Table 4 backward-compatibility).
- **Full run:** `kaggle_colab_runner.ipynb` on Kaggle or Colab; `DATA_DIR = None` triggers the fallback chain, otherwise a manual path overrides. `resolve_data_dir` walks up to depth 5 to find `KiMoRe/` or `UI-PRMD/`.
- **Partial run (no Gemini needed to backfill the [NEXT-RUN] fields):** Cells 1–5 and 8 of the notebook — Setup, Config, Data, Features, Detector, Report — populate `per_flag_subject_split` on Table 1 and `detector_vs_clinician_kappa` on Table 4 without touching Gemini. Cells 6 (Agreement) and 7 (Hallucination) can be skipped on this pass.
- **Static slide exhibits:** `PYTHONPATH=. python scripts/make_slide_exhibits.py` — writes three PNGs under `slides_figures/`, no network.
- **Determinism:** `RANDOM_SEED` in `src/config.py`; all splits, calibration search, and bootstrap resampling are seeded.
- **Audit trail:** `DECISIONS.md` logs every pivot, dated, append-only.

---

## 8. References (to be formatted)

- Capecci, M. et al. *The KIMORE dataset: KInematic assessment of MOvement and clinical scores for remote monitoring of physical REhabilitation.* IEEE TNSRE, 2019.
- Vakanski, A. et al. *A data set of human body movements for physical rehabilitation exercises.* Data, 2018 (UI-PRMD).
- Panni, A. S. et al. *Agreement between pose-estimation and marker-based systems for knee flexion in squats.* (verify ref)
- Parmar, P. & Morris, B. *Fitness-AQA.* (verify ref)
- Zhang, Y. et al. *FormCoach: VLM hallucination in exercise critique.* arXiv:2508.07501.

---

*End of draft. Numbers in §4 come from `results_bundle (4)/`; `[NEXT-RUN]` fields appear automatically on the next notebook execution (Cells 1–5 + 8 are sufficient; no Gemini needed).*
