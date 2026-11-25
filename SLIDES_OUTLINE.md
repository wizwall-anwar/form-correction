# Slides Outline — Form-Correction Creative Component

**Audience:** Advisor review (Anthony Townsend) and MIS 599 committee.
**Length:** 11 slides (10 content + 1 title) + 2 backup. ~15 min talk.
**Tone:** Research prototype. Every number is either verified or explicitly marked as pending/synthetic.
**Visual rules:** Every table built from synthetic data carries `SYNTHETIC — not evidence`. Every rotated-view slide carries `synthetic view`. No injury, risk, diagnose, treat language anywhere on the deck.

---

## Slide 1 — Title

**Content:**
- Title: *A Phone-Only, Abstaining Form-Feedback Prototype for the Squat*
- Subtitle: *Clinician-Score Benchmark, Synthetic Multi-View Degradation, and a Grounded VLM Critique*
- Anwar Ayoon · MIS 599 Creative Component · Advisor: Anthony Townsend · Ivy College of Business · 2026-11-25

**Speaker notes:** Open with the one-line problem — can a phone-only pipeline measure squat form *honestly*, including knowing when to refuse? Signal up front that the research backbone is intentional: a calibrated, abstaining detector plus a grounded critique, not a product pitch.

---

## Slide 2 — Problem and framing

**Content:**
- Published squat form-quality systems lean on IMUs, depth sensors, multi-cam.
- Deployable question: *what can a phone measure honestly, and where must it refuse?*
- Three design rules, enforced in code and in every caption:
  1. Flags are measurements, not injury verdicts.
  2. VLM vs clinician = *model-vs-clinician reliability*, never validity.
  3. Rotated skeletons are SYNTHETIC views, not real cameras.

**Speaker notes:** These are not stylistic preferences — they are the discipline that lets the paper make the small set of claims it does make. Point to CLAUDE.md as the operational copy of these rules. This is also why a VLM-based coach is *not* the core method.

---

## Slide 3 — Architecture at a glance

**Figure:** Horizontal 5-box flow — `m00_data → m01_features → m02_detector → m03_agreement → m04_report` with `app/biocoach_demo` branching off `m02`. Arrow annotations: *subject-safe splits*, *synthetic rotation 0°–60°*, *grounded verifier*, *refusal gate*.

**Content:**
- Five small modules, each with a `--smoke-test` under 5 s offline.
- KIMORE Ex5 primary → UI-PRMD fallback → biomechanical synthetic last; data source stamped on every artifact.
- Constrained demo turns verified numbers into 3 PT-style bullets, or refuses.

**Speaker notes:** Emphasize smallness. Every file has a plain-English docstring; no clever tricks. The owner (me) must be able to read and explain every line — that is a hard rule for the creative component, not an afterthought.

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
- 75 subjects / 75 sequences — one rep per subject.
- Scores compressed in the top quartile (0.83–1.00).
- Caveats shown as bullets, not footnotes:
  - **UI-PRMD** segment-to-landmark mapping is anatomical, not name-equivalent; "incorrect" reps are intentional, not natural errors.
  - **Clip-vs-subject leakage comparison is untestable here** (1 seq / subject).

**Speaker notes:** The score clustering is the most important fact on this slide — it is why we moved from 3-way classification to regression. Call out the earlier tertile bug (72/3/0 split, 0.96 accuracy artifact) honestly: it was caught, the fix is in `train_tertile_edges`, and that run's numbers are void. Owning the bug is part of the research backbone.

---

## Slide 5 — Methodology pivot: classification → regression

**Content:**
- Scores compressed near 1.0 → a 3-way label discards signal.
- **Primary metric:** Spearman ρ and MAE on raw scores, held-out subjects, subject-bootstrap 95% CIs, train-mean MAE baseline.
- **Classification kept only for** calibration (temperature scaling on disjoint val group) and abstention (Table 3).
- Earlier `_score_bin` bug: quantile *probabilities* used as score cutoffs → fixed to tertile edges computed on **training subjects only**.

**Speaker notes:** This is the methodological pivot most likely to come up in review. Frame it as: when the data tells you the task is regression, you let it. Tertile classification does not disappear — it moves to where it adds value (calibration + a confidence threshold for abstention), not where it hides signal.

---

## Slide 6 — Verified result: physiological flag extraction

**Figure:** Left panel — a schematic stick figure with the three flags labeled (knee depth, trunk lean, symmetry). Right panel — a line plot of left vs right knee flexion across the rep for KIMORE ex5 subject 0, peak annotated at ≈ 99–111°.

**Content (VERIFIED):**
- Knee flexion on KIMORE ex5 subject 0 peaks at ≈ **99–111°** with left ≠ right — physiological for a deep squat.
- Validates the pickle's XYZ-column choice (cols 4:7 of each `(T, 7)` joint cell); angle is axis-order-invariant, so a plausible permutation couldn't produce the healthy curve by accident.
- **Caught knee-depth bug:** old code took min knee flexion (= standing moment, ≈ 0°); fixed to max (deepest point), renamed `knee_flexion_min_deg → knee_flexion_max_deg` across m01, m02, biocoach, and tests.

**Speaker notes:** This slide carries two messages. First, the pipeline actually extracts correct physiology on the real licensed data — not synthetic. Second, the bug we caught (`min` vs `max`) is exactly the kind of silent error that makes a detector look fine on toy data and fail on real data; catching it was the reward for insisting on real-data inspection before demo polish.

---

## Slide 7 — Synthetic multi-view degradation (0° → 45°)

**Figure:** `fig1_multi_view_degradation.png` (to generate on next Colab run) — two y-axes, same x (view angle 0°/15°/30°/45°/60°): left y = mean flag-error in degrees, right y = Spearman ρ on held-out subjects. Caption in-figure: *SYNTHETIC rotated views, not real camera views.*

**Content:**
- Rotation about vertical compresses sagittal knee geometry and makes L–R symmetry ambiguous beyond ~45°.
- Table 2 (per-angle Spearman ρ, MAE, mean flag-error) rebuilt on each Colab run; captions carry `is_synthetic_view=True`.
- Degradation curve is the point: the pipeline must *know* a view is bad.

**Speaker notes:** This is where the architecture pays off. We can't claim synthetic rotation behaves like real off-plane camera footage — no occlusion, no pose-estimator error, no clothing. We can claim the geometric lower bound: even in the best case where you only lose projection, the flags degrade this fast. Real cameras can only be worse, which strengthens the abstention argument.

---

## Slide 8 — Calibration, abstention, and the grounded verifier

**Figure (left):** `fig1_risk_coverage.png` — selective risk vs coverage curve.
**Figure (right):** A 2-column "fabricated vs verified" demo — red X on *"Knee depth measured at 999.0 degrees"* (wrong number → rejected) and *"Knee depth measured at 82.4 degrees; injury note"* (banned term → rejected); green check on the template bullet that replaces both.

**Content (VERIFIED):**
- Temperature scaling on a disjoint validation subject group; ECE reported in Table 3.
- `selective_accuracy_at_coverage` replaces the earlier `precision_at_coverage` empty-dict output.
- `verify_claims` + banned-term filter:
  - Fabricated 999.0° → rejected (±0.05° tolerance against detector values).
  - Banned term `injury` → rejected.
  - Both rejections are unit-tested in `_smoke_test`.
- Refusal path: `abstain_flag=True` → exactly one sentence, no critique.

**Speaker notes:** This is the demo's safety story. The verifier is not a sentiment filter — it's a numerical cross-check against the detector's own measurements, plus a term blocklist. Together with the refusal path, this is what keeps the demo honest when the detector is uncertain or when a downstream model (even a well-intentioned VLM paraphrase) tries to put a number in the output that the detector never produced.

---

## Slide 9 — VLM vs clinician agreement [PENDING]

**Table placeholder (do not fill with 999.0 / 888.0 stubs):**

| View (synthetic) | Cohen's κ | Quadratic-weighted κ | n_scored | n_stub |
|---|---|---|---|---|
| 0° | **PENDING-M03** | PENDING-M03 | PENDING-M03 | PENDING-M03 |
| 45° | **PENDING-M03** | PENDING-M03 | PENDING-M03 | PENDING-M03 |

**Content:**
- Gemini currently `503 UNAVAILABLE` under high demand → table deliberately held open.
- m03 now has: model-list fallback walk, `Retry-After` honored, exponential backoff + jitter, result caching by `(seq_id, angle, seed)`, `STUB — no result` excluded from κ (never substituted).
- Table 4 will carry subject-bootstrap CIs, Fleiss' κ for self-consistency across repeated runs, and a detector-vs-clinician κ comparison row.
- Hallucination rates currently in Table 4 are a *verifier control* (planted free-form string), not a VLM measurement — noted in the caption.

**Speaker notes:** Be explicit that the empty cells are a feature, not a gap. Filling them with planted stub values would be exactly the kind of fabrication the verifier on the previous slide rejects. The resilient-retry work on m03 is what I did this week in response to the 503s — it is real engineering, not a workaround for missing numbers.

---

## Slide 10 — Pivots log (and why each one made the paper more honest)

**Content:**
- **Classification → regression** — scores too compressed; classification was discarding signal.
- **Hand-rolled numpy over scikit-learn (local)** — `import sklearn` alone cost 5–7 s, blew the hard 5 s smoke-test budget.
- **Three-way fit/val/test subject split** — earlier cell calibrated on test subjects; now asserted disjoint.
- **UI-PRMD Table 2 (Kinect, 22 joints) over Kinect v2 25-joint** — verified via paper, not assumption.
- **`resolve_data_dir()` made recursive (depth 5)** — Colab's `data/` wrapper and Kaggle's mount prefix silently fell through to synthetic without it.
- **`selective_accuracy_at_coverage` replacing `precision_at_coverage`** — the old output was an empty dict.
- **m03 Gemini resilience** — model-list fallback, `Retry-After`, backoff, caching; stubs excluded from κ.

**Speaker notes:** Pivots are the research. Each line here is a bug or wrong assumption that would have produced a confident, publishable-looking number that was quietly wrong. This slide is the answer to *"why has the design changed since October?"* — and the answer is that each change moved the paper from *looks right* toward *is right*.

---

## Slide 11 — What this prototype can and cannot claim

**Two columns:**

**Can claim**
- Flag-based detector behavior under subject-level splits on KIMORE Ex5.
- Degradation of flags and score prediction in synthetic rotated views.
- Whether calibrated abstention buys selective accuracy at matched coverage.
- Whether the grounded verifier lowers unsupported-claim rates on template critique.

**Cannot claim**
- Clinician inter-rater agreement (KIMORE ships a combined score).
- That rotated skeletons behave like real off-plane cameras.
- Transfer to phone video (Kinect-to-phone untested).
- That any flag means injury risk or incorrect form for any person.
- That a VLM is a qualified evaluator.
- That UI-PRMD "incorrect" reps are natural errors.

**Speaker notes:** Close on the discipline, not the demo. This is the slide the committee will remember. Next steps: run BlazePose on KIMORE RGB to probe Kinect-to-phone transfer; re-open the clinician-vs-clinician question if KIMORE per-rater scores surface; and complete Table 4 once Gemini throughput stabilizes.

---

## Backup slide A — Reproducibility

- Repo: `github.com/wizwall-anwar/form-correction`.
- Smoke tests: `python -m src.m00_data --smoke-test` (and m01–m04, biocoach), each < 5 s, offline.
- Full run: `kaggle_colab_runner.ipynb` on Kaggle or Colab; `DATA_DIR = None` triggers the fallback chain; `resolve_data_dir` walks up to depth 5.
- Determinism: `RANDOM_SEED`, seeded splits, seeded bootstrap, seeded calibration search.
- Audit trail: `DECISIONS.md` append-only, dated.

**Speaker notes:** Use if asked how a reviewer or future student would re-run the evaluation end-to-end.

---

## Backup slide B — Data-schema verification trail (DECISIONS.md §10)

- UI-PRMD Table 2 order is Kinect/Brekel 22-joint, **not** Kinect v2 25-joint; paper + Zenodo archive verified.
- KIMORE pickle `ex5` DataFrame schema: no authoritative source found; XYZ column slice and row-index-as-subject-id remain *empirical* (validated by physiological knee geometry, §4.1 of paper).
- Row-based IDs assume one row = one subject; flagged as unverified; switch if KIMORE ever ships a true subject column.

**Speaker notes:** Use if asked about dataset provenance or why the parser warnings exist in the loader.
