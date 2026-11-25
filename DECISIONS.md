# DECISIONS.md — audit trail

Running log of what we decided, why, and where we stand. Append new entries. Do not rewrite old ones.
Owner: Anwar Ayoon. Advisor: Anthony Townsend (MIS 599). Last updated: 2026-11-25.

## 1. Status snapshot

| Item | Status |
| --- | --- |
| Feasibility report (Claude Project doc) | Done 2026-10-26. Roadmap re-centered prototype-first 2026-11-24. |
| Scope: presentable prototype plus research backbone | Decided (comment on the report doc, 2026-11-24). **Advisor sign-off not yet confirmed.** |
| Part 1 / Part 2 / Part 3 framing | Locked 2026-11-24 (section 2). |
| Repo scaffold | Done 2026-11-24. src/, app/, tests/, results/, requirements.txt, README.md. |
| All 7 src/app modules + kaggle_colab_runner.ipynb | Done 2026-11-24, via Codex Dispatch. All `--smoke-test` flags pass offline, under 5s each (see section 9). |
| Data access (KIMORE Ex5, Princeton 7-mistake labels, UI-PRMD) | Real datasets downloaded and loaders rewritten against them 2026-11-25 (KIMORE pickle at `data/KiMoRe/`, UI-PRMD "Correct" files at `data/UI-PRMD/Correct/`; "Incorrect" pending extraction). KIMORE's joint data is verified; UI-PRMD's exact joint-column order is not (see section 5, 2026-11-25). Princeton labels still a stub (`check_princeton_labels` always returns `None`). |
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
- **2026-11-24**: Wired real KIMORE Exercise 5 and UI-PRMD deep-squat loading into `m00_data.py` (via Codex Dispatch), on top of the existing manual -> fetch-stub -> synthetic chain. `config.py` gained `resolve_data_dir()` (checks an explicit arg, then `FORM_CORRECTION_DATA_DIR` env var, then common Colab/Kaggle paths, `Path.exists()` only — no network, no listing) plus KIMORE score-column candidates and UI-PRMD correct/incorrect labels. `_parse_kimore` expects `<group>/<subject>/Es5/Raw/*.csv` (per-frame `Joint_X/Y/Z` columns) with a sibling `Es5/Label/*.csv` holding a `ClinicalScore`/`TotalScore`/`Score` column; `_parse_uiprmd` expects `<subject>/(Positions|Angles)/(Correct|Incorrect)/*squat*.txt` with >=30 numeric columns. Both are best-effort against the datasets' commonly documented layouts (never verified against an actual unzipped copy, since this machine never downloads data per CLAUDE.md rule 2) — if a real download's column names or folder depth differ, the parser logs a warning per subject and skips it rather than crashing; worst case the whole dataset falls through to synthetic, same as today. `load_data()` was fixed to actually try UI-PRMD when KIMORE returns nothing real, matching D2's KIMORE-primary/UI-PRMD-fallback/synthetic-last order (it previously only ever called `load_kimore_ex5`, so the documented fallback was dead code). `.gitignore` gained `KiMoRe/` and `UI-PRMD/` so an unzip into the repo root can't get committed by accident. 32/32 tests pass.
- **2026-11-25**: The owner downloaded real KIMORE and UI-PRMD files and the 2026-11-24 layout guesses above turned out wrong for both. Claude directly inspected the real files (plain Python, not the project pipeline — CLAUDE.md rule 3 still only allows `--smoke-test` to run the actual modules locally) and replaced `_parse_kimore`/`_parse_uiprmd` with parsers matching the VERIFIED real formats:
  - **KIMORE**: the owner's copy is a single pickle (`kimore_exercise_dataset.pkl`), not per-subject CSVs. It's a dict with keys `ex1`..`ex5`; `ex5` (squat) is a pandas DataFrame, one row per sequence, 25 lowercase joint-name columns (e.g. `spinebase`, `kneeleft`) plus a `cTS` clinician-score column. Each joint cell is a `(T, 7)` array — columns 0:4 are a unit quaternion (verified: norm == 1.0), columns 4:7 are XYZ position; only the position is used. **The DataFrame has no subject-id column at all** — the row's integer index is used as `subject_id` (`kimore_ex5_row_{i}`), assuming one row = one unique subject/session. If KIMORE ever ships a real subject column, switch to it — the row-index assumption is a placeholder, not a verified fact. Unpickling needs `pandas`; added it to requirements.txt but import it lazily inside the parsing function only, so `--smoke-test` (which never touches this file) stays under 1s, confirmed via `grep -n "^import pandas\|^from pandas" src/m00_data.py` returning nothing.
  - **UI-PRMD**: real files are `UI-PRMD/(Correct|Incorrect)/Kinect/Positions/m{MM}_s{SS}_e{EE}_positions.txt`, comma-delimited, no header, 66 numeric columns (22 joints x XYZ). Only movement `m01` is used (`UI_PRMD_SQUAT_MOVEMENT_CODE` in config.py) — confirmed via an independent web search (two sources) that movement 1 is Deep Squat in UI-PRMD's documented 10-movement list, since the dataset's own site (`webpages.uidaho.edu/ui-prmd`) now 404s and MDPI/PMC blocked automated fetches of the paper. **The exact column-to-joint-name order inside those 66 columns could not be verified anywhere** (tried the dataset site, the MDPI publisher page, PMC, and a GitHub/HuggingFace search for an existing parser — all failed or returned nothing useful). Per CLAUDE.md's "pick the most reasonable default, log it, continue" rule: the parser maps the first 30 of the 66 columns into our 10 standardized joint slots *positionally*, with no identity claim, and logs a warning whenever UI-PRMD samples are actually produced, saying their joint identity is unverified and flags from them shouldn't be trusted the way KIMORE's named-joint flags are. Low practical risk today since UI-PRMD is already the documented fallback behind KIMORE (D2) and only gets used if KIMORE's pkl is missing or fails to parse. **Revisit if the real joint order ever surfaces** (the paper, a parser someone else wrote, or just empirically — e.g. plot a known-squat episode's "knee" slot and check it shows a big range of motion).
  - Real dataset files now live at `data/KiMoRe/kimore_exercise_dataset.pkl` and `data/UI-PRMD/Correct/...` (gitignored, never committed). `data/UI-PRMD/Incorrect/...` is pending — the owner's download was still extracting as of 2026-11-25; move it into place with the same `Kinect/Positions/m{MM}_s{SS}_e{EE}_positions.txt` shape once it finishes.
  - 32/32 tests pass (new fixtures build tiny fake pickles/text files matching the real formats, not the real data files). Smoke test still <1s.
- **2026-11-25**: `resolve_data_dir()` made recursive (bounded `os.walk`, depth cap 5) so it returns the directory containing `KiMoRe/`/`UI-PRMD/` however deeply it's nested, with zero manual edits on either platform. The owner's real uploads were nested deeper than the old code checked: Colab at `/content/drive/MyDrive/form-correction-data/data/KiMoRe` (an extra `data/` wrapper from uploading the whole `data/` folder) and Kaggle at `/kaggle/input/datasets/<user>/<name>/data/KiMoRe` (Kaggle's own mount prefix plus the same wrapper). The old code only checked exact fixed paths and one level under `/kaggle/input`, so both silently fell through to synthetic — which is exactly how a `[Data] synthetic` / shallow-angle result slips in unnoticed. The walk stops at the first match and never recurses past depth 5, so it stays fast.
- **2026-11-25**: Fixed a real knee-depth bug in `kinematic_flags` (`m01_features.py`). It took the **minimum** knee flexion across the rep — which is the standing moment (~0 deg), not the squat depth — so `knee_depth_pass` could never be True (dead logic). Switched to the **maximum** flexion (the deepest point) and renamed the key `knee_flexion_min_deg` -> `knee_flexion_max_deg` across `m01_features.py`, `m02_detector.py`, `app/biocoach_demo.py`, and the tests; updated ARCHITECTURE.md section 4 wording to match. Trunk-lean and symmetry were already correct and untouched.
  - **Verification on the real local pkl** (offline data inspection, not a pipeline run — same kind of check used to confirm the pickle format): KIMORE ex5 subject 0 peaks at ~99-111 deg knee flexion with left != right, which is physiological for a deep squat. This **also validates KIMORE's XYZ column order** (columns 4:7 of each `(T,7)` joint cell are the real position) — the coordinate-order doubt flagged in section 10 is resolved for the knee geometry. The angle between joint vectors is axis-order-invariant anyway, so a plausible XYZ permutation couldn't have produced the healthy curve by accident.
  - **The 29.1 deg / identical-left-right reading** that surfaced during testing was NOT a real-data problem: it was the verification plot accidentally running on synthetic fallback data (synthetic subject 0 computes to exactly 29.1/29.1, matching the reading). Root cause was the plot's `resolve_data_dir(None)` hitting the nested-folder path bug above and silently getting synthetic data, fixed by the recursive `resolve_data_dir` change.
  - **Minor known imprecision, not fixed:** `KINECT_V2_TO_STANDARD_JOINT_MAP` folds both `AnkleLeft` and `FootLeft` into the single `ankle_left` slot (no foot slot exists), and in the parser the foot overwrites the ankle, so the "ankle" used in the knee-flexion vector is actually the foot position (shifts the real-data peak from ~108 deg to ~99 deg). Acceptable for a prototype; if finer knee accuracy is wanted, make the parser prefer the true `AnkleLeft`/`AnkleRight` joint over the collapsed foot.

- **2026-11-25 (first real Colab run, KIMORE Ex5, 75 rows) — what it showed and what changed.**
  - **Score spread (local read of the pkl, not a pipeline run):** `cTS` is min 0.50, q1 0.83, median 0.89, q3 0.96, max 1.00, std 0.10, 37 distinct values. All five exercises (ex1-ex5) have the same `cTS` distribution, so no other exercise has wider variance and switching exercises would not help. The identical distribution suggests `cTS` is a per-subject total score repeated across exercises (consistent with, but not proof of, one row per subject).
  - **Root cause of the 72/3/0 bins was a bug, not the data.** `_score_bin` used the quantile probabilities (1/3, 2/3) as absolute score cutoffs; true tertiles of `cTS` are about 0.86 and 0.94 and give 24/24/27. The old accuracy 0.96, the leakage comparison and the abstention curve from that run are void. Fixed: `CLINICIAN_SCORE_BIN_QUANTILES` in config, `assign_score_bins` in `m00_data.py` (dataset-level, used only for VLM kappa, which trains nothing), and train-subjects-only edges in `m02_detector.py` for the detector. Datasets with under 3 distinct scores (UI-PRMD binary) use rank bins.
  - **Primary evaluation is now regression on the raw score:** ridge on the 3 flag measurements, reported as Spearman rho and MAE on held-out subjects, with subject-bootstrap 95% CIs and the train-mean MAE as a constant baseline so we can see whether the flags beat "predict the average". Reason: scores are compressed near the top, so a class label throws away signal. Tertile classification is kept only for calibration and abstention (table3), where it is now balanced. The median-split fallback was not needed.
  - **Evaluation was leaky in the notebook, now fixed.** Old cell 10 calibrated on the test subjects and scored all 75 samples including training ones. `run_evaluation` now uses train / validation / test subject groups (validation for calibration), asserts all three pairs are disjoint, and computes metrics on test subjects only.
  - **Clip-vs-subject leakage comparison is not informative on this data.** 75 subjects, 75 sequences: each subject has one clip, so the clip split cannot leak. Table 1 now says so (`max_sequences_per_subject`). Do not report it as "leakage did not matter"; it was untestable.
  - **Table 3** `precision_at_coverage` was an empty dict; replaced by `selective_accuracy_at_coverage` (accuracy among the most confident fraction kept, not class precision).
  - **Table 2** `flag_error` is now one mean over the three flags (knee, trunk, symmetry in degrees) instead of per-flag values; per-flag errors from the earlier run were knee ~2.2, trunk ~0.9, symmetry ~3.5 deg at 15-60 deg views. Restore per-flag if the paper needs it.
  - **Gemini was never called, for two reasons.** (1) Notebook cell 4 swallowed every secret-read exception, so a missing or disabled Colab secret left `VLM_BACKEND=placeholder` silently. Cell 4 now prints the exception class and a hint, and never prints the key. (2) `gemini-2.0-flash-lite` was shut down 2026-06-01 (Google deprecations page); `GEMINI_MODEL_ID` is now `gemini-3.1-flash-lite`. The key now goes in the `x-goog-api-key` header, not the URL, so error text cannot leak it. Failed calls are counted (`n_stub`) and excluded from kappa; they are never replaced by placeholder bins. The `AQ.`-prefixed key format is untested against the live API.
  - **Hallucination rates in table4 are a verifier control, not a measurement.** The 999.0 / 888.0 free-form string is planted in notebook cell 14 (not in m03). `hallucination_rates["free_form"]` therefore counts a fabricated string by construction; table4 now carries a note saying so. A real free-form-vs-grounded comparison needs actual VLM free-form text and is not built yet.
  - **Tooling:** Codex default model `gpt-6-sol` in `~/.codex/config.toml` is not in this ChatGPT account's catalog, so dispatch failed with `model_not_found`. Dispatches now pass `--model gpt-5.6-sol` per call (verified working); the config file was left untouched. Fix the config default if you want plain dispatches to work.
  - **Verification state:** 53/53 pytest (needs `--basetemp` on this machine because the default Windows temp dir is access-denied in the sandbox) and all six `--smoke-test` runs under 2 s. Nothing in this entry has been run on real data yet: the new tables come from the next Colab run.

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

## 10. 2026-11-25 — Dataset schema verification (Firecrawl)

- The [UI-PRMD paper, Tables 1 and 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5773117/) identifies `m01` as deep squat and gives separate ordered lists for Vicon positions (39 markers), Vicon angles (39 joints), and Kinect positions and angles (22 joints). Each entry has three coordinates, so Kinect position/angle rows have 66 values and Vicon rows have 117. Position coordinates are in millimeters and angles in degrees. The order is **not** Kinect v2's 25-joint order. The Kinect order is: Waist, Spine, Chest, Neck, Head, Head tip, Left Collar, Left Upper arm, Left forearm, Left hand, Right collar, Right upper arm, Right forearm, Right hand, Left upper leg, Left lower leg, Left foot, Left leg toes, Right upper leg, Right lower leg, Right foot, Right leg toes. The loader uses the Table 2 positions list only and maps Waist/Spine/Collar/upper leg/lower leg/foot to the ten standardized landmarks. These segment-to-landmark equivalents are anatomical approximations; there is no exact name equivalence in the paper. Vicon angles and positions are not treated as interchangeable with Kinect positions.
- The [Zenodo archive and README](https://zenodo.org/records/23104965) verify comma-delimited text, complete and segmented archives, `m01_s##_e##_positions` episode naming, and `_inc` for intentionally suboptimal episodes. The loader accepts the archive folder names (`Segmented Movements` and `Incorrect Segmented Movements`) and the earlier `Correct`/`Incorrect` mirror layout, requires exactly 66 finite numeric values per row, and keeps all repetitions of a subject under one split key. The Zenodo record includes direct archives and a separate reduced Vicon-angle dataset with quality scores; those scores must not be silently substituted for raw Kinect binary labels. [OpenDataLab](https://opendatalab.com/OpenDataLab/UI-PRMD/download) is another listed mirror, but its page did not verify internal file layout.
- The [KIMORE paper abstract](https://pubmed.ncbi.nlm.nih.gov/31217121/) verifies 78 subjects (44 healthy, 34 with motor dysfunction) and clinician questionnaire scores, but does **not** document `kimore_exercise_dataset.pkl`, subject-ID columns, `c_score`/`f_score`/`cTS`, score direction, or the seven-value joint-cell order. The [KiMoRe wrapper](https://github.com/petteriTeikari/KiMoRe_wrapper) describes Kinect v2 joint trajectories and groups, but is an alternate HDF5/MAT representation. A [third-party pickle loader](https://github.com/Hieuvu4438/ADAPT-Rehab/blob/main/evaluation/benchmarks/kimore.py) reports an `ex5` DataFrame, 25 joint arrays and `cTS`, while a [probe from the same repo](https://github.com/Hieuvu4438/ADAPT-Rehab/blob/main/scripts/probe_kimore.py) contradicts that loader about whether XYZ occupies columns 0:3 or 4:7. Neither is an authoritative pickle schema. The present KIMORE parser's XYZ slice and row-index subject IDs remain **unverified**; do not use its clinical agreement results until a licensed local copy confirms coordinate order, unique subject identities, score direction, and score field meaning. Row-based IDs must not be assumed leakage-safe if multiple rows can belong to one subject.
- UI-PRMD is a binary, intentionally performed movement benchmark, not a replacement for KIMORE clinician-score agreement. On UI-PRMD fallback, clinician-vs-model agreement claims must be suppressed or explicitly marked unavailable.
- For reproducible schema checks, Table 2's **Vicon position** order (1–39) is: LFHD, RFHD, LBHD, RBHD, C7, T10, CLAV, STRN, RBAK, LSHO, LUPA, LELB, LFRM, LWRA, LWRB, LFIN, RSHO, RUPA, RELB, RFRM, RWRA, RWRB, RFIN, LASI, RASI, LPSI, RPSI, LTHI, LKNE, LTIB, LANK, LHEE, LTOE, RTHI, RKNE, RTIB, RANK, RHEE, RTOE. The distinct **Vicon angle** order (1–39) is: Head (absolute), Left head, Right head, Left neck, Right neck, Left clavicle, Right clavicle, Thorax (absolute), Left thorax, Right thorax, Pelvis (absolute), Left pelvis, Right pelvis, Left hip, Right hip, Left femur, Right femur, Left knee, Right knee, Left tibia, Right tibia, Left ankle, Right ankle, Left foot, Right foot, Left toe, Right toe, Left shoulder, Right shoulder, Left elbow, Right elbow, Left radius, Right radius, Left wrist, Right wrist, Left upperhand, Right upperhand, Left hand, Right hand. These lists are documentation only; the loader does not parse Vicon data.
