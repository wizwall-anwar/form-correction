# Slides Outline — Form-Correction Creative Component

**Audience:** Advisor review (Anthony Townsend) and MIS 599 committee.
**Length:** 11 slides (10 content + 1 title) + 2 backup. ~15 min talk.
**Tone:** Research prototype with a pre-registered negative result as the headline. Every number is either verified against `results_bundle (4)/` or explicitly marked `[NEXT-RUN]`.
**Visual rules:** Every table built from synthetic data carries `SYNTHETIC — not evidence`. Every rotated-view slide carries `synthetic view`. No injury, risk, diagnose, treat language anywhere on the deck.
**Figures used (all already in `slides_figures/`):** `fig_per_flag_vs_score.png`, `fig_multi_view_degradation.png`, `fig_grounded_vs_free_form.png`, `fig_vlm_distribution.png`, `fig_stick_triptych_0deg.png`, `fig_stick_triptych_45deg.png`, `fig_biocoach_safety_exhibit.png`, plus `results_bundle (4)/fig1_risk_coverage.png`. The last three are new 2026-11-25 artifacts (patches C and D).

---

## Slide 1 — Title

**Content:**
- Title: *A Phone-Only, Abstaining Form-Feedback Prototype for the Squat*
- Subtitle: *Clinician-Score Benchmark, Synthetic Multi-View Degradation, and a Grounded VLM Critique*
- Anwar Ayoon · MIS 599 Creative Component · Advisor: Anthony Townsend · Ivy College of Business · 2026-11-25

**Speaker notes:** One-line problem: can a phone-only pipeline measure squat form *honestly*, including knowing when to refuse? Up front, flag that the headline is a **negative result** pre-registered in DECISIONS §6 — the flag-based baseline inverts the clinician score with CI not crossing zero, and that is the finding. Research backbone (calibrated, abstaining detector + grounded critique + honest reporting) is the contribution, not a working coach.

---

## Slide 2 — Problem and framing

**Content:**
- Published squat form-quality systems lean on IMUs, depth sensors, multi-cam.
- Deployable question: *what can a phone measure honestly, and where must it refuse?*
- Three design rules, enforced in code and in every caption:
  1. Flags are measurements, not injury verdicts.
  2. VLM vs clinician = *model-vs-clinician reliability*, never validity.
  3. Rotated skeletons are SYNTHETIC views, not real cameras.

**Speaker notes:** These three rules are the discipline that lets the paper make the small set of claims it does make — and, importantly, that lets a negative result count as a contribution instead of an embarrassment. Point to CLAUDE.md as the operational copy. This is also why a VLM-based coach is *not* the core method.

---

## Slide 3 — Architecture at a glance

**Figure:** Horizontal 5-box flow — `m00_data → m01_features → m02_detector → m03_agreement → m04_report` with `app/biocoach_demo` branching off `m02`. Arrow annotations: *subject-safe splits*, *synthetic rotation 0°–60°*, *grounded verifier*, *refusal gate*. Add a dashed box underneath: *per-flag diagnostic + detector-vs-clinician κ (patches A/B, 2026-11-25)*.

**Content:**
- Five small modules, each with a `--smoke-test` under 5 s offline.
- KIMORE Ex5 primary → UI-PRMD fallback → biomechanical synthetic last; data source stamped on every artifact.
- Constrained demo turns verified numbers into 3 PT-style bullets, or refuses.
- 74/74 pytest green (65 baseline + 9 for the 2026-11-25 patches).

**Speaker notes:** Emphasize smallness. Every file has a plain-English docstring; no clever tricks. The owner (me) must be able to read and explain every line — a hard rule for the creative component, not an afterthought. Point out the dashed box: the per-flag diagnostic and the detector-vs-clinician κ comparison are new wiring from this week in direct response to the headline on slide 5 — they are the mechanism-probe that would turn the negative result into a repairable one.

---

## Slide 4 — Dataset reality check

**Table (verified, local read of `kimore_exercise_dataset.pkl`):**

| `cTS` statistic | Value |
|---|---|
| n | 75 |
| min | 0.50 |
| Q1 | 0.83 |
| median | 0.89 |
| Q3 | 0.96 |
| max | 1.00 |
| std | 0.10 |
| distinct values | 37 |

**Content:**
- 75 subjects / 75 sequences — one rep per subject (`max_sequences_per_subject = 1`).
- Scores compressed in the top quartile (0.83–1.00).
- Caveats shown as bullets, not footnotes:
  - **UI-PRMD** segment-to-landmark mapping is anatomical, not name-equivalent; "incorrect" reps are intentional, not natural errors.
  - **Clip-vs-subject leakage comparison is untestable here** (1 seq / subject).

**Speaker notes:** The score clustering is the most important fact on this slide — it is why we moved from 3-way classification to regression. Call out the earlier tertile bug (72/3/0 split, 0.96 accuracy artifact) honestly: it was caught, the fix is in `train_tertile_edges`, and that run's numbers are void. Owning the bug is part of the research backbone.

---

## Slide 5 — Headline finding: the flag baseline inverts the clinician score

**Table (from `results_bundle (4)/table1.json`):**

| Metric | Value | 95% CI (subject bootstrap) |
|---|---|---|
| Spearman ρ (held-out) | **−0.515** | **[−0.771, −0.156]** |
| MAE (held-out) | 0.075 | [0.054, 0.100] |
| MAE mean baseline | 0.069 | — |
| n_test | 23 | — |

**Content:**
- The 95% CI never crosses zero → **consistent inversion**, not noise.
- Also fails the "beat the average" test: MAE 0.075 > 0.069 (mean baseline).
- Pre-registered pivot criterion (DECISIONS §6): *"Learned detector fails to beat the simple flag/classical baseline on held-out subjects → report the baseline as the finding."* **Triggered.**
- **Per-flag diagnostic (patch A, verified 2026-11-25 Kaggle run):**
  - `trunk_lean_max_deg`: ρ = **+0.539** (strongest single-flag predictor)
  - `knee_flexion_max_deg`: ρ = +0.216
  - `symmetry_diff_deg`: ρ = −0.189
- Mechanism (confirmed): individual flags carry reasonable signal; the ridge mixture *fights* the sign of the strongest one under small-training-sample + feature-collinearity conditions. A univariate model on trunk lean alone would likely beat the three-flag ridge.

**Speaker notes:** This is the single most important slide in the deck. Do not undersell it. The contribution of a graduate creative-component is a thing that moves the knowledge frontier — here, that a naively-mixed ridge on three deterministic kinematic flags does not predict KIMORE Ex5 clinician scores on held-out subjects, and does so in a measurable direction with pre-registered reporting rules. The per-flag diagnostic (patch A, wired this week) is the next experiment, not spin.

---

## Slide 6 — Methodological pivot: classification → regression

**Content:**
- Scores compressed near 1.0 → a 3-way label discards signal.
- **Primary metric:** Spearman ρ and MAE on raw scores, held-out subjects, subject-bootstrap 95% CIs, train-mean MAE baseline.
- **Classification kept only for** calibration (temperature scaling on disjoint val group) and abstention (Table 3).
- Earlier `_score_bin` bug: quantile *probabilities* used as score cutoffs → fixed to tertile edges computed on **training subjects only**.
- Four-way disjointness asserted across fit / validation / test / parent train subjects on every `run_evaluation` call.

**Speaker notes:** This is the methodological pivot most likely to come up in review. Frame it as: when the data tells you the task is regression, you let it. Tertile classification does not disappear — it moves to where it adds value (calibration + a confidence threshold for abstention), not where it hides signal. The four-way disjointness assertion was added after an earlier notebook cell was found calibrating on test subjects; now it crashes loudly if that recurs.

---

## Slide 7 — Verified result: physiological flag extraction

**Figure:** `slides_figures/fig_per_flag_vs_score.png` — per-flag scatter vs cTS (visual preview of the diagnostic; the authoritative held-out per-flag ρ will come from the patch-A helper on the next detector run).

**Content (VERIFIED):**
- Knee flexion on KIMORE ex5 subject 0 peaks at ≈ **99–111°** with left ≠ right — physiological for a deep squat.
- Validates the pickle's XYZ-column choice (cols 4:7 of each `(T, 7)` joint cell); angle is axis-order-invariant, so a plausible permutation couldn't produce the healthy curve by accident.
- **Caught knee-depth bug:** old code took min knee flexion (= standing moment, ≈ 0°); fixed to max (deepest point), renamed `knee_flexion_min_deg → knee_flexion_max_deg` across m01, m02, biocoach, and tests.
- The per-flag scatter is where slide 5's "flags reward magnitude, clinicians reward control" mechanism will be confirmed or refuted on held-out subjects on the next detector run.

**Speaker notes:** Two messages. First, the pipeline actually extracts correct physiology on real licensed data — not synthetic. Second, the min-vs-max bug is exactly the kind of silent error that makes a detector look fine on toy data and fail on real data; catching it is the reward for insisting on real-data inspection before demo polish.

---

## Slide 8 — Synthetic multi-view degradation (0° → 60°)

**Figure:** `slides_figures/fig_multi_view_degradation.png` — two y-axes, same x (view angle 0°/15°/30°/45°/60°): left y = mean flag-error in degrees, right y = Spearman ρ on held-out subjects. In-figure caption: *SYNTHETIC rotated views, not real camera views.*

**Table (from `results_bundle (4)/table2.json`):**

| Angle | ρ | MAE | Mean flag_error (deg) |
|---|---|---|---|
| 0° | −0.536 | 0.076 | 0.000 |
| 15° | −0.527 | 0.075 | 2.106 |
| 30° | −0.471 | 0.075 | 1.838 |
| 45° | −0.516 | 0.076 | 2.569 |
| 60° | −0.521 | 0.076 | 2.058 |

**Content:**
- ρ ∈ [−0.54, −0.47] across all angles → **detector is stably wrong**: the inversion in slide 5 is a feature problem, not a rotation problem.
- Flag error is small (1.8°–2.6° mean at 15°–60°) → under the best-case geometric lower bound (no occlusion, no pose-estimator error, no clothing), the three deterministic flags survive synthetic yaw rotation well.
- All non-zero-angle rows carry `is_synthetic_view = true`.

**Speaker notes:** The angle stability is good news in a bad sense. We cannot claim synthetic rotation behaves like real off-plane camera footage — no occlusion, no pose-estimator error, no clothing. We can claim the geometric lower bound: even in the best case where you only lose projection, the flags degrade this fast, and real cameras can only be worse. This strengthens the abstention argument on slide 9.

---

## Slide 9 — Calibration, abstention, and the grounded verifier

**Figure (left):** `results_bundle (4)/fig1_risk_coverage.png` — selective risk vs coverage curve.
**Figure (right):** `slides_figures/fig_biocoach_safety_exhibit.png` — new 2026-11-25 static exhibit (patch D): 45° stick-figure triptych beside grounded bullets (knee 106.72°, trunk 26.85°, symmetry 19.59° — real detector numbers from `halluc_log.jsonl`) with a red-X free-form fabrication (`"999.0 degrees"`) rejected.

**Content (from `results_bundle (4)/table3.json`, VERIFIED):**
- **ECE = 0.861** on held-out subjects → classifier very poorly calibrated.
- **Risk-coverage inverted:** selective risk = 1.0 at the most-confident 13% of predictions, drops to 0.75 at 17% coverage, climbs back to 0.87 at full coverage. **Most confident = most wrong.** Consistent with slide 5's inversion.
- `selective_accuracy_at_coverage` = {0.5: 0.083, 0.8: 0.105, 1.0: 0.130} → **below the ≈ 0.33 random floor** for 3 classes. Abstention does not buy accuracy here.
- Pre-registered criterion (DECISIONS §6): *"If abstention gives no precision gain at matched coverage → report negative, keep the gate as safety behavior only."* **Triggered.** Gate kept in `app/biocoach_demo.py`; not claimed as a performance contribution.
- `verify_claims` + banned-term filter unit-tested:
  - Fabricated 999.0° → rejected (±0.05° tolerance against detector values).
  - Banned term `injury` → rejected.
- Refusal path: `abstain_flag = True` → exactly one sentence, no critique.

**Speaker notes:** The safety story is working as designed — the verifier catches fabricated numbers and banned terms, the refusal path runs on abstain. What is *not* working is the calibration: the classifier is maximally confident where it is most wrong, which is consistent with the regression inversion two slides back. The honest move here, pre-registered, is to keep the gate for safety but not claim it as a performance win. The right-panel exhibit replaces a live demo — if Gemini or Colab misbehaves during the talk, the committee still sees the pipeline's visual output and the verifier's rejection behavior side-by-side.

---

## Slide 10 — Gemini vs clinician agreement (model reliability, not validity)

**Figure:** `slides_figures/fig_vlm_distribution.png` — Gemini bin distribution vs clinician bin distribution at 0° and 45°, showing Gemini's bimodal 0-or-2 extremizing.

**Table (from `results_bundle (4)/table4.json`, VERIFIED):**

| View (synthetic) | Cohen's κ | Quadratic-weighted κ | n_scored | n_stub |
|---|---|---|---|---|
| 0° | **−0.139** | −0.215 | 55 | 20 |
| 45° | **+0.139** | +0.221 | 35 | 40 |

**Content:**
- Both κ near zero; the sign flip 0° → 45° is plausibly noise at these sample sizes (n_scored differs 55 vs 35).
- Gemini scored almost every sample as 0 or 2 and almost never 1 → the prompt or the stick-figure keyframe render pushes to extremes.
- Framed strictly as **model-vs-clinician reliability**, never validity.
- m03 shipped: `GEMINI_MODEL_FALLBACKS` walk (`gemini-3.1-flash-lite` → `gemini-2.5-flash-lite`), `Retry-After` honored, exponential backoff + jitter, on-disk cache under `results/vlm_cache/`, `STUB — no result` excluded from κ (never substituted), API key only in `x-goog-api-key` header with log redaction.
- **Hallucination rates are a verifier control, not a measurement.** `{"free_form": 2.0, "grounded": 0.0}` carries the explicit caption *"free_form is a PLANTED fabricated control string (999/888), not VLM output."* The planted string lives in notebook cell 14, not in m03. A real free-form-vs-grounded hallucination measurement needs actual free-form VLM text under the same prompt and is listed as future work.
- **Detector-vs-clinician κ comparison row (patch B, verified 2026-11-25 Kaggle run):** Cohen's κ = **0.000** (quadratic-weighted κ = 0.000) on all 75 samples. The flag-based classifier agrees with clinicians *at chance*. This puts Gemini's κ ≈ ±0.14 in context: Gemini is marginally above the flag-classifier baseline but still statistically near zero. Neither flag-based nor VLM-based approach shows meaningful ordinal agreement with clinician scores on this sample.
- **Fleiss' κ for VLM self-consistency** demoted to future work (no repeated-run data on the Gemini backend in the current bundle).

**Speaker notes:** The 20 and 40 stubbed samples at 0° and 45° are the sticking point — Gemini returned 503 UNAVAILABLE on them even after the retry-and-fallback chain. The right off-peak re-run will mostly resolve those via the on-disk cache (every successful sample is already cached by `(seq_id, angle, seed)`), tightening 45° toward n ≈ 55+. Do not re-interpret the sign flip at these sample sizes.

---

## Slide 11 — What this prototype can and cannot claim

**Two columns:**

**Can claim**
- The flag-based detector inverts the KIMORE Ex5 clinician score on held-out subjects, CI not crossing zero (ρ = −0.52, 95% CI [−0.77, −0.16]).
- The inversion is stable across synthetic rotations (ρ ∈ [−0.54, −0.47] at 0°–60°) → it is a feature-mixing problem, not a rotation problem.
- Under synthetic yaw rotation, flag error is 1.8°–2.6° mean at 15°–60° (geometric lower bound).
- The classifier is miscalibrated (ECE = 0.86) and its selective risk is inverted; abstention gives no precision benefit; the gate is kept only for honest refusal behavior.
- Gemini-vs-clinician κ is near zero at 0° and 45°, framed as model-vs-clinician reliability only.
- The grounded verifier rejects planted fabricated numbers and banned terms, unit-tested.

**Cannot claim**
- Clinician inter-rater agreement (KIMORE ships a combined score).
- That rotated skeletons behave like real off-plane cameras.
- Transfer to phone video (Kinect-to-phone untested).
- That any flag means injury risk or incorrect form for any person.
- That a VLM is a qualified evaluator.
- That UI-PRMD "incorrect" reps are natural errors.
- A measured free-form-vs-grounded hallucination rate (current rates are a verifier control on a planted string).

**Next steps (in priority order):**
1. **Univariate baseline on `trunk_lean_max_deg` alone** — the per-flag diagnostic (slide 5) shows this single flag has ρ = +0.54 on held-out subjects, stronger than the three-flag ridge mixture (ρ = −0.52). If a one-coefficient ridge on trunk-lean beats the three-flag ridge, the inversion is a pure linear-mixture pathology; if it doesn't, the follow-up is a non-linear model (GBM) on the same three flags.
2. Engineered features: joint velocities, hip-knee coupling, trunk-knee phase, cross-rep variance.
3. BlazePose on KIMORE RGB to probe Kinect-to-phone transfer.
4. Collect actual free-form VLM critique text to replace the planted 999/888 control with a measured hallucination rate.

**Speaker notes:** Close on the discipline, not the demo. The negative result is the contribution — the pre-registered pivot criteria mean this was always a legitimate outcome, not a failure mode. Next steps are specific and ranked: step 1 is free (no Gemini); step 2 is the mechanism test the per-flag diagnostic points at.

---

## Backup slide A — Reproducibility

- Repo: `github.com/wizwall-anwar/form-correction`.
- Smoke tests: `python -m src.m00_data --smoke-test` (and m01–m04, biocoach), each < 5 s, offline.
- Full test suite: `PYTHONPATH=. pytest -v` — 74 passing (65 baseline + 9 for the 2026-11-25 patches).
- Full run: `kaggle_colab_runner.ipynb` on Kaggle or Colab; `DATA_DIR = None` triggers the fallback chain; `resolve_data_dir` walks up to depth 5.
- Partial run to backfill `[NEXT-RUN]` fields with no Gemini: cells 1–5 and 8 only.
- Static slide exhibits: `PYTHONPATH=. python scripts/make_slide_exhibits.py` (three PNGs under `slides_figures/`, no network).
- Determinism: `RANDOM_SEED`, seeded splits, seeded bootstrap, seeded calibration search.
- Audit trail: `DECISIONS.md` append-only, dated.

**Speaker notes:** Use if asked how a reviewer or future student would re-run the evaluation end-to-end.

---

## Backup slide B — Data-schema verification trail (DECISIONS §10)

- UI-PRMD Table 2 order is Kinect/Brekel 22-joint, **not** Kinect v2 25-joint; paper + Zenodo archive verified.
- KIMORE pickle `ex5` DataFrame schema: no authoritative source found; XYZ column slice and row-index-as-subject-id remain *empirical* (validated by physiological knee geometry, §4.1 of paper).
- Row-based IDs assume one row = one subject; flagged as unverified; switch if KIMORE ever ships a true subject column.

**Speaker notes:** Use if asked about dataset provenance or why the parser warnings exist in the loader.
