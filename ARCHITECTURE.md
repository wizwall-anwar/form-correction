# ARCHITECTURE.md — plain-English blueprint

Goal: every file Codex writes should be easy to read and explain. Read this before dispatching any task.
Decisions and limits live in @DECISIONS.md. Hard rules live in @CLAUDE.md.

## 1. The idea in one paragraph

Take squat skeleton sequences with clinician scores. Compute simple, named joint-angle flags (knee depth, trunk lean, left-right symmetry). Make synthetic off-plane views by rotating the 3D skeleton and adding jitter. Train a small detector that predicts the clinician score bin, calibrate it, and let it abstain when unsure. Test whether a VLM agrees with clinicians, and how that changes from 0 to 45 degrees. A demo then turns verified numbers into a 3-bullet critique, or refuses when the detector abstains.

## 2. Repo layout

```
form-correction/
  CLAUDE.md
  DECISIONS.md
  ARCHITECTURE.md
  README.md                   short: how to run smoke test and the notebook
  PAPER_DRAFT.md              working manuscript, synced to the latest Kaggle run
  SLIDES_OUTLINE.md           talk outline, synced to the paper
  requirements.txt
  kaggle_colab_runner.ipynb   cloud entry point (section 6)
  src/
    __init__.py
    config.py                 all thresholds, seeds, angles, bins, timeouts
    m00_data.py               load data, build Sample objects, fallback chain
    m01_features.py           rotate, jitter, project, compute flags
    m02_detector.py           splits, baseline, calibration, abstain, per-flag diagnostic
    m03_agreement.py          VLM-vs-clinician kappa, detector-vs-clinician kappa
    m04_report.py             tables, figure, manifest, results_bundle.zip
  app/
    __init__.py
    biocoach_demo.py          3-bullet grounded critique, refusal, hallucination log
  scripts/
    make_slide_exhibits.py    offline static-exhibit renderer (patch D, 2026-11-25)
  tests/
    test_m00_data.py ... test_m04_report.py
    test_biocoach_demo.py
    test_leakage.py
    test_config.py
  slides_figures/             static exhibits for the talk; not written by the pipeline
  results/                    written on Kaggle/Colab, not committed
    manifest.json  table1..table4  fig1_risk_coverage.png  weights/  halluc_log.jsonl
```

Why `m00_...` and not `00_...`: Python cannot import a file whose name starts with a digit.

## 3. Data schema

One `Sample` per recorded sequence (dataclass in `m00_data.py`):

| Field | Meaning |
| --- | --- |
| `subject_id` | Person. Splits key on this. Never split below subject level. |
| `session_id` | Recording session, if the dataset has it. |
| `seq_id` | Unique sequence id. |
| `exercise` | `"squat"`. |
| `joints` | Array `[T, J, 3]`, standardized joint set (hips, knees, ankles, spine base/mid, shoulders). |
| `fps` | Frames per second. |
| `clinician_score` | Float, or `None`. Raw dataset score. |
| `clinician_bin` | 0/1/2 ordinal bin from config tertiles. |
| `flags_3d` | Reference flags from the unrotated skeleton (section 4, m01). |
| `view_deg` | Rotation applied. 0 for original. |
| `is_synthetic_view` | `True` whenever `view_deg != 0` or jitter was added. |
| `data_source` | `manual`, `fetched`, or `synthetic`. Stamped on every output. |

Joint mapping: Kinect v2 25-joint skeleton to the standardized set lives in `config.py`.

## 4. Modules

### m00_data.py — get the data
- Fallback chain, 30 s timeout per step: `DATA_DIR` manual path, then Zenodo/GitHub fetch, then synthetic.
- `load_kimore_ex5()`: KIMORE Exercise 5 squats plus clinician scores.
- `load_uiprmd()`: fallback. Binary correct/incorrect labels.
- `check_princeton_labels()`: one 15 s check. Returns labels or `None`. Never blocks.
- `make_synthetic_squats(n_subjects, seed)`: parametric squat generator with controllable depth, lean, asymmetry. Score derived from those parameters. Stamped `SYNTHETIC - not evidence`.
- Writes `results/manifest.json` with the source that ran.
- `--smoke-test`: builds 4 tiny dummy samples, checks schema. No network.

### m01_features.py — views and flags
- `rotate_view(joints, deg)`: rotate about the vertical axis. `deg=0` returns the input unchanged.
- `add_jitter(joints, sigma, seed)`: seeded Gaussian noise.
- `project_2d(joints)`: orthographic projection to the image plane.
- `kinematic_flags(joints)`: deterministic, returns measured values and boolean flags:
  - knee depth: maximum knee flexion angle reached (the deepest point of the rep) and whether it passes the depth threshold
  - trunk lean: maximum trunk angle from vertical
  - symmetry: left-right knee angle difference
- Thresholds come from `config.py`, tuned on train subjects only. Flags are measurements, not verdicts.
- `flag_error(flags_view, flags_3d)`: error in degrees versus the 3D reference. This is the flag-error-vs-angle curve.
- `--smoke-test`: rotation at 0 is identity, flags are deterministic on dummy data.

### m02_detector.py — learn, calibrate, abstain, per-flag diagnostic
- `make_splits(samples, seed)`: grouped by `subject_id`. Asserts `len(set(train_subj) & set(test_subj)) == 0`.
- `make_clip_split(samples, seed)`: deliberately leaky split used only to show the clip-vs-subject gap.
- Primary (regression): closed-form ridge on the three flag measurements predicts the raw clinician score. Reported as Spearman rho and MAE on held-out subjects, with subject-bootstrap 95% CIs and train-mean MAE baseline.
- Secondary (classification): multinomial logistic regression over train-only tertile bins; used only for calibration and abstention.
- Calibration: temperature scaling, fit on a held-out validation subject group disjoint from both fit and test.
- Abstention: threshold on calibrated confidence. Output per sample: `pred_bin`, `confidence`, `abstain_flag`.
- `risk_coverage(...)`: selective risk versus coverage. Feeds `fig1`.
- `per_flag_correlations(samples)`: Spearman rho of each flag vs the clinician score on the supplied samples (patch A, 2026-11-25). `run_evaluation` emits it under `per_flag_subject_split` (computed on held-out subjects); `build_table1` surfaces it. Diagnoses whether individual flags carry signal that the mixed ridge model cancels out.
- `run_evaluation(...)`: asserts four-way disjointness across fit, validation, test, and the parent train subject groups; returns regression metrics, per-flag correlations, classifier calibration, risk-coverage, and per-angle table2 rows.
- `--smoke-test`: trains on 8 dummy tensors for a few steps, under 5 s.

### m03_agreement.py — VLM vs clinician, detector vs clinician
- For each view angle, render keyframes (start, bottom, end) as stick figures from the rotated and projected skeleton.
- Ask the VLM for a score bin. Backend set in `config.py`, 30 s timeout. On timeout or exhausted retries, mark `STUB - no result`. Never fake a number.
- Resilience: model-list fallback walk (`GEMINI_MODEL_FALLBACKS`), `Retry-After` honored, exponential backoff with jitter bounded by `VLM_RETRY_MAX_ATTEMPTS`/`VLM_RETRY_BASE_DELAY_S`/`VLM_RETRY_MAX_DELAY_S`, on-disk cache under `results/vlm_cache/` keyed by `(seq_id, angle, seed)`. API key sent only in the `x-goog-api-key` header; log output is redacted.
- Cohen's kappa (quadratic-weighted and unweighted): VLM vs clinician bin, at 0 and 45 degrees.
- `detector_vs_clinician_kappa(detector_bins, clinician_bins)` (patch B, 2026-11-25): Cohen's kappa (unweighted and quadratic-weighted) plus `n_scored` for the flag-based classifier against clinician bins. Surfaced in `build_table4` as `detector_vs_clinician_kappa`. Gives the VLM kappa a non-zero baseline to be compared against.
- Fleiss' kappa for VLM self-consistency across repeated runs is future work: no repeated-run data on the Gemini backend is collected by the current runner.
- Wording rule: output is labeled "model-vs-clinician reliability". Never "validity".
- `--smoke-test`: kappa on toy ratings with a known answer. Offline stub backend.

### m04_report.py — package results
- `table1`: data source, subject and sequence counts, subject-split vs clip-split baseline; optional `per_flag_subject_split` field (patch A kwarg) carries per-flag Spearman rho on held-out subjects.
- `table2`: detector performance and flag error by view angle (0, 15, 30, 45, 60). Captions say synthetic views; non-zero-angle rows carry `is_synthetic_view=True`.
- `table3`: calibration and abstention: ECE, `selective_accuracy_at_coverage` (not `precision_at_coverage`), selective risk curve, test class counts.
- `table4`: VLM-vs-clinician kappa (0 vs 45) and hallucination rates (free-form vs grounded); optional `detector_vs_clinician_kappa` field (patch B kwarg) carries the flag-based classifier's Cohen's kappa against clinician bins as a comparison line.
- `fig1_risk_coverage.png`.
- Saves model weights, `manifest.json`, then zips everything to `results_bundle.zip`.
- Stamps `SYNTHETIC - not evidence` on any output built from synthetic data.
- Backward-compatible: `build_table1`/`build_table4` accept their new kwargs as optional; omitted kwargs leave the field absent.
- `--smoke-test`: builds all tables from dummy inputs, checks files exist.

### app/biocoach_demo.py — the grounded critique
- Input: one detector output record (flag values, `abstain_flag`, `confidence`, `view_deg`).
- If `abstain_flag=True`: print one line saying the view is unreliable and give no critique.
- Otherwise print exactly 3 bullets: named deviation plus its measured number, in plain PT-style language.
- Default verbalizer is a fixed template, so it is deterministic. An optional VLM paraphrase is allowed only behind the verifier below.
- `verify_claims(bullets, detector_out)`: every number and every deviation name in the bullets must match the detector output within tolerance. Otherwise the bullet is rejected and replaced by the template.
- Banned-term filter: injury, hurt, damage, risk, diagnose, treat, and similar. Any hit rejects the text.
- Hallucination log (`halluc_log.jsonl`): for the same inputs, record free-form VLM output and grounded output. Count claims unsupported by the detector facts. Rates feed `table4`.
- `--demo`: runs on a bundled sample JSON. `--smoke-test`: runs on dummy detector output, offline.

## 5. Tests (written and run by Codex)

- `test_leakage.py`: `len(set(train_subj) & set(test_subj)) == 0` on every split used in evaluation.
- Schema and shape tests for every module.
- Rotation at 0 degrees is identity. Same seed gives the same jitter.
- Flags are deterministic. Known toy squat gives known flag values.
- Kappa on toy data matches hand-computed values.
- Abstention: lowering the threshold never increases coverage.
- Verifier rejects a planted fabricated number and a planted injury word.
- Fallback chain: mocked timeouts fall through in order, never hang.
- Each `--smoke-test` finishes in under 5 s with no network.

## 6. kaggle_colab_runner.ipynb

Cell 0 (markdown) carries an "Expected result" banner stating that ρ < 0 with a CI not crossing zero is the published negative finding, not a bug (patch C, 2026-11-25). Cells in order:

1. Setup: install requirements, set seeds, clone or `git pull --ff-only` the repo from GitHub.
2. Config: top-level `DATA_DIR = None` hook; `resolve_data_dir()` walks up to depth 5 to find `KiMoRe/` or `UI-PRMD/`. Loads Gemini API key from Colab or Kaggle secrets.
3. Data: run the fallback chain. Print which source ran.
4. Features: build views 0/15/30/45/60 with jitter.
5. Detector: splits, regression, calibration, abstain; also prints `per_flag_subject_split` (patch A) and `detector_vs_clinician_kappa` (patch B) inline.
6. Agreement: VLM-vs-clinician kappa (served from on-disk cache if present).
7. Hallucination comparison via the demo module.
8. Report: tables (with the new patch A/B fields populated), figure, weights, `results_bundle.zip`. Download link at the end.

**Partial run to backfill diagnostic fields without touching Gemini:** cells 1–5 and 8 only. Useful when Gemini is flaky and you just need `per_flag_subject_split` and `detector_vs_clinician_kappa` in the bundle.

## 7. scripts/make_slide_exhibits.py

Offline plain-Python exhibit renderer (patch D, 2026-11-25). Reads the KIMORE pickle when present (falls back to the synthetic generator when it's absent), renders start/bottom/end stick-figure keyframes at 0° and 45° using `render_keyframes_image`, and composes a side-by-side `fig_biocoach_safety_exhibit.png` with the grounded BioCoach bullets next to a rejected free-form fabrication. Defaults to reading `results/halluc_log.jsonl` record 1 for the detector numbers, with a baked-in fallback. Mirrors the DECISIONS §5 (2026-11-25) offline-inspection pattern: no pipeline run, no network, no training.

Every cell in the notebook prints a one-line status. A failed cell must not stop later cells that can still run on fallback data.
