# DECISIONS.md — audit trail

Running log of what we decided, why, and where we stand. Append new entries. Do not rewrite old ones.
Owner: Anwar Ayoon. Advisor: Anthony Townsend (MIS 599). Last updated: 2026-11-24.

## 1. Status snapshot

| Item | Status |
| --- | --- |
| Feasibility report (Claude Project doc) | Done 2026-10-26. Roadmap re-centered prototype-first 2026-11-24. |
| Scope: presentable prototype plus research backbone | Decided (comment on the report doc, 2026-11-24). **Advisor sign-off not yet confirmed.** |
| Part 1 / Part 2 / Part 3 framing | Locked 2026-11-24 (section 2). |
| Repo scaffold | Done 2026-11-24. src/, app/, tests/, results/, requirements.txt, README.md. |
| All 7 src/app modules + kaggle_colab_runner.ipynb | Done 2026-11-24, via Codex Dispatch. All `--smoke-test` flags pass offline, under 5s each (see section 9). |
| Data access (KIMORE Ex5, Princeton 7-mistake labels, UI-PRMD) | Unverified. Loaders are stubbed fallback chains (manual -> fetch-stub -> synthetic); no real dataset wired up yet. |
| Kaggle/Colab full run | Not started. Notebook scaffolded with all 8 cells, not executed (requires cloud environment per CLAUDE.md). |
| Demo (`app/biocoach_demo.py`) | Done 2026-11-24. Deterministic template, verifier, banned-term filter, hallucination log — all implemented and tested. |

## 2. Locked decisions

**D1. Deliverable.** A presentable squat form-feedback prototype, with a research backbone that keeps it graduate-level: a calibrated, abstaining detector and an honest evaluation. The app is the demo. The evaluation is the research.

**D2. Part 1 framing: Clinician-Score Benchmark + Synthetic Multi-View Degradation + VLM-vs-Clinician Agreement Study.**
- Primary data: KIMORE Exercise 5 (squatting), with clinician scores.
- Named deviations come from deterministic kinematic joint-angle flags: knee depth, trunk lean, left-right symmetry.
- Princeton 7-mistake KIMORE labels: run one 15 s availability check. If gated or unavailable, proceed immediately without them.
- Fallback dataset: UI-PRMD. Last resort: biomechanical synthetic generator.
- Off-plane degradation: rotate the 3D skeleton to 0, 15, 30, 45, 60 degrees and add Gaussian jitter.
- Agreement: Cohen's and Fleiss' kappa across 0 vs 45 degrees. Framed strictly as **model-vs-clinician reliability**. We never claim a VLM equals human PT ground truth.

**D3. Part 2 framing: constrained BioCoach interface.**
- `app/biocoach_demo.py` prints a 3-bullet PT-style critique.
- It verbalizes only detector-verified numbers and named deviations. Zero injury claims.
- When `abstain_flag=True` it refuses to guess and says the view is unreliable.
- Hallucination-rate logging compares free-form VLM output against grounded output.

**D4. Part 3 execution: cloud-first.**
- `kaggle_colab_runner.ipynb` with a top-level `DATA_DIR` manual-path hook, then automated Zenodo/GitHub fetch, then synthetic fallback.
- Exports `table1` to `table4`, `fig1_risk_coverage.png`, and model weights into `results_bundle.zip`.

**D5. Tooling.** Claude Code orchestrates. Codex Dispatch writes all Python, math, training loops, and pytest suites. No heavy work or dataset download on the local laptop. Only a `<5s` `--smoke-test` on tiny dummy tensors runs locally.

**D6. No human recruiting or filming for now.** The original P2 plan (human experts label filmed participants, then kappa) is **deferred, not dropped**. Revisit if labelers and IRB clearance become available.

## 3. Pivot log

- **2026-10-26** Feasibility report written. Recommended: validity-grounded, uncertainty-aware squat detector. First experiment: human inter-rater kappa.
- **2026-11-24** A comment on the report doc said this is not a study and a presentable prototype is wanted. Roadmap re-centered prototype-first (thin demo by week 4).
- **2026-11-24** No external PT recruiting or live filming available. Original P2 reframed to the D2 design above.
- **2026-11-24** Execution moved from the web chat to Claude Code plus Codex Dispatch, with training on Kaggle/Colab.

## 4. What this design can and cannot show

**Can show**
- How a detector trained on clinician scores behaves under subject-level splits, with a clip-split comparison to expose leakage.
- How named geometric flags and score predictions degrade as the view moves off-plane (geometrically, in synthetic views).
- Whether calibrated abstention buys precision at matched coverage.
- Whether a VLM's ratings agree with clinician scores, and how that changes from 0 to 45 degrees.
- Whether grounded output cuts unsupported claims relative to free-form VLM output.

**Cannot show (do not claim)**
- That experts agree on what correct form is. KIMORE gives one combined score per sequence. Per-rater scores may not be available. Clinician-vs-clinician agreement is therefore not measured here.
- That rotated skeletons behave like real off-plane camera footage. Rotation adds no occlusion, no pose-estimator error, no clothing effects.
- That results transfer to phone video. KIMORE skeletons are Kinect-derived. Phone pose estimators have their own error.
- That any flag means injury risk or incorrect form for a person. Flags are measurements.
- That a VLM persona is a qualified evaluator.
- That UI-PRMD "incorrect" reps are natural errors. They were performed on purpose by healthy subjects.
- That "hallucination" means false in the real world. It means unsupported by detector-verified facts.

## 5. Defaults chosen by the architecture (not locked; change by logging here)

- Kappa needs categories. Default: bin clinician scores into 3 ordinal bins by tertile, fixed before any results are seen. Report quadratic-weighted Cohen's kappa and unweighted kappa.
- Fleiss' kappa: used across repeated VLM runs/personas for self-consistency, reported separately from VLM-vs-clinician Cohen's kappa.
- Confidence intervals: bootstrap resampling by subject.
- VLM input: rendered stick-figure keyframes (start, bottom, end) of the rotated and projected skeleton. Offline stub used in smoke tests.
- Flags are computed from the rotated and orthographically projected 2D coordinates, and compared with 3D reference flags from the unrotated skeleton. That gives a measurable flag-error-vs-angle curve.
- Abstention: calibrate (temperature or isotonic), then selective prediction on confidence. Report risk-coverage.
- **2026-11-24**: `m02_detector.py`'s baseline is logistic regression on the 3 kinematic flag measurements only (no temporal TCN/GRU) — ARCHITECTURE.md marks the temporal model as "only kept if it earns its place," and the flag baseline is the simplest thing that works for a prototype. Revisit if the flag baseline underperforms badly on real data.
- **2026-11-24**: `m02_detector.py` and `m03_agreement.py` use hand-rolled numpy implementations (logistic regression, temperature scaling, macro-F1, Cohen's/Fleiss' kappa) instead of scikit-learn. Reason: on the dev machine, `import sklearn` alone costs 5-7s cold, which blew the hard 5-second `--smoke-test` budget (CLAUDE.md rule 3) regardless of how fast the actual computation was. Dropped `scikit-learn`, `scipy`, `pandas`, and `torch` from requirements.txt since nothing currently imports them. Revisit if a real (non-toy) training run on Kaggle/Colab needs scikit-learn's maturity — that's fine there since it isn't time-boxed to 5s.
- **2026-11-24**: Repo hygiene pass. `.gitignore` was missing `results/` (ARCHITECTURE.md section 2 says it's written on Kaggle/Colab, not committed); added it, plus `*.tar`/`*.tar.gz` alongside the existing `*.zip` rule. `kaggle_colab_runner.ipynb`'s Setup cell never actually cloned the repo on Colab — it only added `/content/form-correction` to `sys.path` and assumed it existed, so a fresh Colab runtime would hit `ModuleNotFoundError` on `from src.config import RANDOM_SEED`. Fixed by cloning `https://github.com/wizwall-anwar/form-correction.git` (or `git pull --ff-only` if already present) before the path insert and pip install. README.md's merge-conflict markers (left over from an earlier merge) were already resolved in the working tree by the time this pass ran; confirmed clean via grep across the whole repo.

## 6. Abandon / pivot criteria

- KIMORE and UI-PRMD both unreachable: run the pipeline on synthetic data for engineering checks only. Label every output `SYNTHETIC - not evidence`. Ask the advisor for access.
- Learned detector fails to beat the simple flag/classical baseline on held-out subjects: report the baseline as the finding.
- Abstention gives no precision gain at matched coverage: report it as a negative result. Keep the abstain gate in the demo as a safety behavior, not a claimed contribution.
- Grounded output does not lower the unsupported-claim rate versus free-form: drop that claim. Keep the app as a demo.
- Clip-split and subject-split numbers match: say so. Leakage did not matter on this data.

## 7. Open items

- [ ] Advisor sign-off that D1 and D2 satisfy the creative-component bar.
- [ ] Does KIMORE ship per-rater clinician scores? (If yes, add a clinician-vs-clinician kappa. That restores part of the original validity map.)
- [ ] Are the Princeton critique labels reachable? (Checked once at run time.)
- [ ] VLM backend and cost ceiling (free or low-cost only).
- [ ] KIMORE / UI-PRMD licensing and access path (request form or direct download).
- [ ] Exact Graduate College deadline. The paper is due 2 weeks before the defense.
- [ ] Stretch: run a phone pose estimator (e.g., BlazePose) on KIMORE RGB to bridge Kinect-to-phone transfer.

## 8. Evidence anchors (from the feasibility report; recheck before citing in the paper)

- MediaPipe knee flexion in squats: mean bias -13.9 deg (SD 15.1), limits of agreement [-43.5, +15.5] deg (Panni et al.). About 2 deg MAE in controlled lateral running (Biomedical Human Kinetics, 2025).
- Fitness-AQA subtle error F-scores: about 53 (knees inward) and about 46 (overhead-press elbow).
- FormCoach VLM hallucination: about 74-97% across models (arXiv:2508.07501).
- Rehab AQA state of the art: Spearman rho about 0.80-0.90 on KIMORE and UI-PRMD.
- UI-PRMD: 10 subjects, 10 movements; incorrect reps are simulated.
