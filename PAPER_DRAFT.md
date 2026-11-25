# A Phone-Only, Abstaining Form-Feedback Prototype for the Squat: Clinician-Score Benchmark, Synthetic Multi-View Degradation, and a Grounded VLM Critique

**Author:** Anwar Ayoon (MIS 599 Creative Component, Ivy College of Business, Iowa State University)
**Advisor:** Anthony Townsend
**Draft date:** 2026-11-25
**Status:** Working manuscript. Verified sections are marked [VERIFIED]. Sections awaiting Gemini throughput are marked [PENDING-M03].

---

## Abstract (placeholder — write last)

One paragraph: we build a phone-only squat form-feedback prototype that (i) predicts a clinician score from three deterministic kinematic flags under subject-level splits, (ii) quantifies how both flags and score prediction degrade when the 3D skeleton is rotated off-plane (synthetic views, 0°–60°), and (iii) delivers a 3-bullet critique that is verified against detector measurements and refuses when confidence is low. VLM-vs-clinician agreement is framed strictly as *model-vs-clinician reliability*, never as validity. We report verified physiological flag extraction on KIMORE Exercise 5 (n = 75), a view-degradation curve across 0°/15°/30°/45°/60°, and a grounded anti-hallucination verifier whose rejection behavior is unit-tested on planted fabrications. The VLM agreement table itself is held open pending stable Gemini throughput.

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

(To expand with citations in final draft; recheck before camera-ready.)
- MediaPipe knee flexion in squats shows mean bias ≈ −13.9° (SD 15.1), with limits of agreement [−43.5°, +15.5°] (Panni et al.). This motivates abstention rather than confident coaching on monocular pose.
- Subtle-error F-scores in Fitness-AQA sit near 53 (knees inward) and 46 (overhead-press elbow) — far from product-grade.
- FormCoach reports VLM hallucination rates of ≈ 74–97% across models on exercise critique tasks (arXiv:2508.07501), motivating the grounded verifier.
- Rehabilitation-AQA state of the art reports Spearman ρ ≈ 0.80–0.90 on KIMORE and UI-PRMD — the target the flag-based baseline is compared against as a skyline.

---

## 2. Architecture

The pipeline is five modules plus a constrained demo, each small enough to read and debug end-to-end.

### 2.1 `m00_data` — ingest with bounded fallbacks

Loads KIMORE Exercise 5 from a verified local pickle; falls back to UI-PRMD; synthetic last. A 30 s timeout per step and a single 15 s availability check for the gated Princeton 7-mistake labels prevent any network stall. The data source actually used is written to `results/manifest.json` and stamped on every downstream artifact.

### 2.2 `m01_features` — deterministic kinematics and synthetic views

`rotate_view(joints, deg)` rotates about the vertical axis; `add_jitter` adds seeded Gaussian noise; `project_2d` is orthographic. `kinematic_flags` returns three measurements:
- **Knee depth** — maximum knee flexion across the rep (deepest point).
- **Trunk lean** — maximum trunk angle from vertical.
- **Symmetry** — maximum absolute left–right knee-angle difference.
Thresholds live in `src/config.py`, tuned on training subjects only. Flags are measurements; `*_pass` booleans are reported but never interpreted as correctness verdicts for a person.

### 2.3 `m02_detector` — subject-safe baseline, calibration, abstention

- **Splits** are grouped by `subject_id`; every evaluation path asserts `len(set(train) & set(test)) == 0`. A deliberately leaky clip-split is kept only for a leakage comparison row.
- **Regression (primary).** Closed-form ridge on the three flag measurements predicts the raw clinician score. Reported as Spearman ρ and MAE on held-out subjects, with subject-bootstrap 95% CIs, and a train-mean MAE baseline so the reader can see whether the flags beat *predict the average*.
- **Classification (secondary).** Multinomial logistic regression over tertile bins; used only for calibration (temperature scaling on a held-out validation subject group) and abstention (threshold on calibrated confidence).
- **Risk-coverage** feeds Figure 1 (`fig1_risk_coverage.png`).

### 2.4 `m03_agreement` — VLM vs clinician as reliability

Keyframes (start, bottom, end) are rendered as stick-figure PNG triptychs from the rotated, projected skeleton. The VLM is prompted for a single-digit bin (0/1/2). Cohen's κ (unweighted and quadratic-weighted) and Fleiss' κ (for self-consistency across repeated runs) are reported, with subject-bootstrap CIs. Reporting language is locked to *model-vs-clinician reliability*. Failed VLM calls return `STUB — no result` and are **excluded** from κ; they are never replaced by a placeholder bin.

Resilience (new, in response to transient `503 UNAVAILABLE` from Gemini): the caller now walks a `GEMINI_MODEL_FALLBACKS` list, honors `Retry-After` headers, and uses exponential backoff with jitter bounded by `VLM_RETRY_MAX_ATTEMPTS`, `VLM_RETRY_BASE_DELAY_S`, and `VLM_RETRY_MAX_DELAY_S`. Successful calls are cached by `(seq_id, angle, seed)`, so a resumed notebook run does not re-spend quota.

### 2.5 `m04_report` — four tables, one figure, one zip

`build_table1` through `build_table4`, `fig1_risk_coverage.png`, `manifest.json`, and `results_bundle.zip`. Any artifact built from synthetic data is stamped `SYNTHETIC — not evidence`. Captions on Table 2 and Table 4 state that the views are synthetic rotated views, not real camera views.

### 2.6 `app/biocoach_demo` — the constrained critique

Input is one detector record (`flags`, `abstain_flag`, `confidence`, `view_deg`).
- If `abstain_flag=True`: exactly one refusal sentence, no critique.
- Otherwise: exactly three bullets from a fixed template.
- `verify_claims` rejects any sentence containing a number that does not match a detector value within ±0.05°, and any sentence containing a banned term (injury, hurt, damage, risk, diagnose, treat, …). The sentence is replaced by the template.
- `log_hallucination_comparison` records unsupported-number counts for free-form text vs grounded bullets, feeding Table 4.

---

## 3. Dataset and Leakage-Free Evaluation

### 3.1 KIMORE Exercise 5 (primary)

The owner's licensed copy is a single pickle, `kimore_exercise_dataset.pkl`, whose `ex5` key is a pandas DataFrame of **75 rows**, one per sequence, with 25 lowercase joint-name columns and a `cTS` clinician-score column. Each joint cell is a `(T, 7)` array whose columns 0:3 are a unit quaternion (verified `‖q‖ = 1.0`) and columns 4:7 are XYZ position; only the position is used. This XYZ-column choice is independently validated by physiological knee geometry (§4.1).

**Score distribution (local read of the pickle, not a pipeline run).** `cTS` is min 0.50, Q1 0.83, median 0.89, Q3 0.96, max 1.00, std 0.10, with 37 distinct values across the 75 sequences. The distribution is identical across all five exercises in the pickle, consistent with (but not proof of) one row per subject.

**Honest limitation — score clustering.** 75% of scores sit at 0.83 or higher; the top quartile is compressed in a 0.04-wide band. Early tertile classification inherited a bug where `_score_bin` used the quantile *probabilities* (1/3, 2/3) as absolute score cutoffs, producing a 72/3/0 split and an artifactual 0.96 accuracy. Fixed: true tertile edges (≈ 0.86 and ≈ 0.94) give balanced 24/24/27 bins, and tertile edges are now computed on **training subjects only** inside `m02_detector`. Datasets with fewer than three distinct scores (e.g. UI-PRMD's binary labels) fall back to rank bins.

**Pivot to regression.** Because scores are compressed near the top of the scale, a 3-way class label discards signal. The primary detector is now ridge regression on the raw score, with tertile classification kept only for calibration and abstention (Table 3). We report Spearman ρ, MAE on held-out subjects, subject-bootstrap 95% CIs, and the train-mean MAE as a constant baseline.

### 3.2 UI-PRMD (fallback) — parent-relative kinematics caveat

UI-PRMD ships binary correct/incorrect labels, and its "incorrect" repetitions were deliberately performed by healthy subjects — not natural errors. Beyond that, the loader uses the paper's Table 2 (Kinect positions, 22 joints × 3 = 66 columns per row); segment names such as *upper leg* / *lower leg* / *foot* are anatomical approximations for our standardized ten-joint slots, and we log a warning whenever UI-PRMD samples are actually produced. Any clinician-vs-model agreement claim is suppressed or marked unavailable when UI-PRMD is the backing dataset.

### 3.3 Princeton 7-mistake labels

One 15 s availability check; if gated or unreachable, the pipeline proceeds without them. Currently returns `None`.

### 3.4 Leakage guard

Every evaluation path asserts `len(set(train_subj) & set(test_subj)) == 0`. The test is in `tests/test_leakage.py` and runs in CI. `run_evaluation` additionally asserts disjointness across the three-way fit/validation/test subject groups introduced after an earlier notebook cell was found calibrating on the test subjects.

**Honest limitation — clip-vs-subject leakage comparison is not informative on this data.** With 75 subjects and 75 sequences (`max_sequences_per_subject == 1`), the clip-split cannot leak by construction. Table 1 says so explicitly. We do not report this as *"leakage did not matter"*; it was untestable on KIMORE Ex5 in its current form.

---

## 4. Verified Results

### 4.1 Physiological flag extraction on KIMORE Ex5 [VERIFIED]

An offline inspection of the real pickle (plain-Python read, not a pipeline run, per CLAUDE.md §3) measured knee flexion on KIMORE ex5 subject 0 at the deepest point of the rep. Values peaked at **≈ 99–111°** with left ≠ right — physiological for a deep squat and consistent with the published range. This result also validates the XYZ-column choice in the pickle (columns 4:7 of each (T, 7) joint cell): a plausible XYZ permutation could not produce the physiological curve by accident, because the inter-vector angle is axis-order-invariant.

A minor known imprecision is logged, not fixed: `KINECT_V2_TO_STANDARD_JOINT_MAP` collapses `AnkleLeft` and `FootLeft` into one `ankle_left` slot (we have no foot slot), and in the parser the foot overwrites the ankle, shifting the knee-flexion peak from ≈ 108° to ≈ 99°. This is acceptable for a prototype; for finer knee accuracy, the parser should prefer the true `Ankle*` joint over the collapsed foot.

We also caught and corrected a knee-depth bug: `kinematic_flags` had taken the *minimum* knee flexion across the rep (the standing moment, ≈ 0°) rather than the maximum (deepest point), so `knee_depth_pass` could never be `True`. Switched to the maximum and renamed `knee_flexion_min_deg` → `knee_flexion_max_deg` across `m01`, `m02`, the demo, and the tests.

### 4.2 Synthetic multi-view degradation (0° → 45°) [VERIFIED, SYNTHETIC VIEWS]

Table 2 reports per-angle Spearman ρ, MAE, and a single mean flag-error (over the three flags) on held-out subjects, at angles 0, 15, 30, 45, and 60 degrees. All non-zero-angle rows carry `is_synthetic_view=True` and the caption reads *"Detector performance and flag error for SYNTHETIC rotated views, not real camera views."*

Rotation about the vertical axis is adversarial for monocular sagittal geometry: the apparent knee-hip-ankle angle becomes compressed as the body turns, and left–right symmetry becomes ambiguous beyond ~45°. The expected and observed pattern is a monotonic increase in mean flag-error with angle, and a monotonic decrease in Spearman ρ. Exact per-angle numbers are regenerated by `kaggle_colab_runner.ipynb` on each Colab run (the earlier run's per-flag errors — knee ≈ 2.2°, trunk ≈ 0.9°, symmetry ≈ 3.5° at 15°–60° views — are not quoted here as evidence, because that run's calibration cell was leaky and has since been fixed).

The degradation curve is the point: a pipeline that does not know it is looking at a bad view will keep coaching confidently against measurements that no longer reflect the real movement. Abstention (§4.3) is the design response.

### 4.3 Calibration, abstention, and grounded critique [VERIFIED]

- **Calibration.** Temperature scaling is fit on a held-out validation subject group disjoint from both fit and test. Reported as ECE in Table 3.
- **Selective prediction.** `risk_coverage` sweeps actual confidence thresholds; Figure 1 (`fig1_risk_coverage.png`) is selective risk vs coverage. Table 3 reports `selective_accuracy_at_coverage` — accuracy among the most-confident top-k — and *not* class precision, because the earlier `precision_at_coverage` output was an empty dict.
- **Anti-hallucination verifier [VERIFIED by unit test].** `verify_claims` and `_contains_banned_term` are unit-tested on two planted fabrications and both are rejected:
  - *"Knee depth measured at 999.0 degrees."* → fabricated number, ≠ any detector value within ±0.05° → replaced by the template.
  - *"Knee depth measured at 82.4 degrees; injury note."* → matches a detector value but contains the banned term `injury` → replaced by the template.
  Both assertions are in `_smoke_test` and run in under 5 s offline.
- **Refusal behavior.** `format_bullets` returns exactly one sentence — *"The view is unreliable, so no critique is given."* — when `abstain_flag=True`, and exactly three measurement sentences otherwise.

### 4.4 VLM-vs-clinician agreement [PENDING-M03]

> **Placeholder.** This table is intentionally held open. Gemini is currently returning transient `503 UNAVAILABLE` under high demand; m03's new fallback-and-retry logic (model-list walk, `Retry-After`, exponential backoff with jitter, caching) is in place to soak up throughput once it stabilizes, but no agreement number is quoted here yet.
>
> We will **never** fill this table with the planted `999.0` / `888.0` values. Those live in notebook cell 14 as a verifier control — they prove the free-form text contains an unsupported number by construction — not as a measurement of VLM behavior. `table4.hallucination_rates["free_form"]` therefore carries a note saying so.
>
> When Gemini throughput returns, the table will contain, with subject-bootstrap CIs:
> - Cohen's κ (unweighted) at 0° and 45°.
> - Quadratic-weighted Cohen's κ at 0° and 45°.
> - Fleiss' κ across repeated runs (self-consistency).
> - A detector-vs-clinician κ row as a comparison line.
> - `n_scored` and `n_stub` counts per cell; `STUB — no result` samples are excluded from κ, never replaced.
> - A free-form-vs-grounded hallucination comparison built from *actual* free-form VLM text, not the planted control.

| View (synthetic rotation) | Cohen's κ | Quadratic-weighted Cohen's κ | n_scored | n_stub |
|---|---|---|---|---|
| 0° | PENDING-M03 | PENDING-M03 | PENDING-M03 | PENDING-M03 |
| 45° | PENDING-M03 | PENDING-M03 | PENDING-M03 | PENDING-M03 |

---

## 5. Discussion

### 5.1 What we can and cannot claim

**Can claim.** How a flag-based detector trained on clinician scores behaves under subject-level splits on KIMORE Ex5; how named geometric flags and score prediction degrade from 0° to 60° in synthetic rotated views; whether calibrated abstention buys selective accuracy at matched coverage; and whether a grounded verifier lowers unsupported-claim rates on template-based critique text.

**Cannot claim.** That clinicians agree on correct form (KIMORE ships a combined score). That rotated skeletons behave like real off-plane camera footage (no occlusion, no pose-estimator error, no clothing). That results transfer to phone video (Kinect-to-phone untested). That any flag means injury risk or incorrect form for any person. That a VLM persona is a qualified evaluator. That UI-PRMD "incorrect" reps are natural errors. That "hallucination" is a claim about real-world truth; here it is solely *unsupported by detector-verified facts*.

### 5.2 Engineering pivots in service of honesty

- **Classification → regression** because scores are compressed near 1.0 and a 3-way label was discarding signal. Classification is retained only for calibration and abstention.
- **Hand-rolled numpy over scikit-learn** because `import sklearn` alone spent 5–7 s cold on the dev machine, blowing the hard 5 s smoke-test budget. The Colab/Kaggle path is free to use scikit-learn.
- **UI-PRMD Table 2 mapping over Kinect v2 25-joint assumption** after Firecrawl verification showed UI-PRMD is 22-joint Brekel order, not Kinect v2 25-joint. Logged in DECISIONS.md §10.
- **Three-way fit/val/test subject split** after an earlier notebook cell was found calibrating on test subjects.
- **`resolve_data_dir()` made recursive** because Colab's extra `data/` wrapper and Kaggle's mount prefix meant the old fixed-path checker silently fell through to synthetic — exactly how a `[Data] synthetic` result slips in unnoticed.

### 5.3 Abandon / pivot criteria (pre-registered)

If the learned detector fails to beat the simple flag baseline on held-out subjects, the baseline is the finding. If abstention gives no precision gain at matched coverage, the negative result is reported and the abstain gate is kept in the demo as a safety behavior only. If grounded output does not lower the unsupported-claim rate versus free-form, that claim is dropped and the app is kept as a demo.

---

## 6. Limitations and Future Work

1. **Kinect-to-phone transfer is untested.** The whole pipeline uses Kinect-derived skeletons; a next step is to run BlazePose (or equivalent) on KIMORE RGB and re-measure the degradation curve.
2. **Score field semantics are not fully documented.** KIMORE's `cTS` has no authoritative schema we could locate; a licensed-copy audit of score direction and per-rater availability remains an open item.
3. **UI-PRMD joint-identity is unverified below the Table 2 slot level.** The parser maps segments to standardized landmarks anatomically, not by name equivalence.
4. **Clip-vs-subject leakage comparison is untestable on KIMORE Ex5 in its current form** (1 sequence per row). A dataset with multiple reps per subject would let us show whether the gap is real on this task.
5. **Hallucination rates are currently a verifier control, not a measurement.** A real free-form-vs-grounded comparison needs actual VLM free-form text, which depends on the m03 agreement table landing.
6. **VLM backend reliability.** Gemini's current `503 UNAVAILABLE` rate under high demand is the immediate blocker for Section 4.4; the mitigations in §2.4 are in place.

---

## 7. Reproducibility

- **Code:** `github.com/wizwall-anwar/form-correction` (modules `m00`–`m04`, `app/biocoach_demo.py`, `kaggle_colab_runner.ipynb`).
- **Smoke tests (local, offline, < 5 s each):** `python -m src.m00_data --smoke-test` and the m01–m04 and biocoach counterparts.
- **Full run:** `kaggle_colab_runner.ipynb` on Kaggle or Colab; `DATA_DIR = None` triggers the fallback chain, otherwise a manual path overrides. `resolve_data_dir` now walks up to depth 5 to find `KiMoRe/` or `UI-PRMD/`.
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

*End of draft. Section 4.4 fills in on the next successful Colab run with Gemini throughput; everything else is ready for advisor review.*
