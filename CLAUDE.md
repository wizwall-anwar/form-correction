# CLAUDE.md

Phone-only squat form-feedback prototype for an MIS 599 creative component (Iowa State).
A calibrated, abstaining detector plus a grounded 3-bullet critique, demoable live.
The research backbone matters more than app polish.

Read first: @DECISIONS.md (why, status, limits) and @ARCHITECTURE.md (layout, modules, schema).
Update DECISIONS.md whenever a decision, pivot, or status changes. Append, dated. Never rewrite history.

## Claims discipline (every file, table, figure, and demo string)
- Never claim a movement flag implies injury risk, or that it is "wrong" for this person. Flags are measurements.
- A VLM is never human PT ground truth. VLM vs clinician is model-vs-clinician reliability only.
- Rotated skeletons are SYNTHETIC views, not real camera views. Set `is_synthetic_view=True` everywhere, including table captions.
- Synthetic fallback data is a pipeline check only. Stamp its outputs `SYNTHETIC - not evidence`.
- KIMORE skeletons come from Kinect. Say so. Transfer to phone video is untested.
- "Hallucination" means a claim not supported by detector-verified facts. It is not a claim about ground truth.

## Execution rules (hard)
1. Delegate to Codex Dispatch: all Python module writing, math, training loops, and pytest suites. Claude Code orchestrates, reviews against ARCHITECTURE.md, and explains. If a module is wrong, re-dispatch with the fix. Do not hand-write it.
2. Never download datasets. Never run training. Never call a network API locally.
3. The only local run allowed is `python -m src.<module> --smoke-test` (and `python -m app.biocoach_demo --smoke-test`) on tiny dummy tensors. It must finish in under 5 seconds, offline.
4. Heavy work runs in `kaggle_colab_runner.ipynb` on Kaggle or Colab.
5. Leakage guard: every evaluation test asserts `len(set(train_subj) & set(test_subj)) == 0`. Split by subject, never by clip or rep. If the assert fails, fix the split. Never weaken the test.
6. Network calls use a 30 s timeout and a fallback chain so nothing stalls:
   `DATA_DIR` manual path -> Zenodo/GitHub fetch -> biomechanical synthetic fallback.
   Record which source ran in `results/manifest.json`.
7. Gated or unreachable data (KIMORE, Princeton 7-mistake labels): one 15 s availability check. If gated, proceed at once with the fallback. Do not wait. Do not ask.
8. Fixed seeds everywhere. Same input, same output.

## Code standards
- Every file starts with a plain-English docstring: what it does, inputs, outputs, why it exists.
- Small functions, type hints, no clever tricks. The owner must be able to read, explain, and debug every file.
- Thresholds live in `src/config.py`, tuned on train subjects only. No magic numbers inline.
- Module files are named `m00_data.py` to `m04_report.py` (digit-leading names cannot be imported).

## Locked scope (details in @DECISIONS.md)
- Data: KIMORE Exercise 5 (squat) clinician scores. UI-PRMD fallback. Synthetic last.
- Named deviations: knee depth, trunk lean, left-right symmetry (deterministic joint-angle flags).
- Views: rotate 3D skeleton 0/15/30/45/60 degrees plus Gaussian jitter. Compare 0 vs 45.
- Agreement: Cohen's kappa (model-vs-clinician reliability only). Fleiss' kappa was planned but is now future work; no repeated-run Gemini data is collected.
- Per-flag diagnostic (Spearman rho of each flag vs cTS on held-out subjects) and detector-vs-clinician Cohen's kappa are reported in `table1` and `table4` respectively (patches A and B, 2026-11-25).
- App: `app/biocoach_demo.py`. 3 bullets, verified numbers only, refuse when `abstain_flag=True`.
- Outputs: `table1`-`table4`, `fig1_risk_coverage.png`, model weights, all in `results_bundle.zip`. Static presentation exhibits (`slides_figures/*.png`) are rendered offline by `scripts/make_slide_exhibits.py` and are NOT part of the bundle.
- Deliverables: `PAPER_DRAFT.md` (synced to the latest bundle) and `SLIDES_OUTLINE.md` (synced to the paper). Treat these as source-of-truth for the committee.

## Out of scope unless DECISIONS.md says otherwise
- Recruiting people, filming participants, new human-labeled data.
- Free-form VLM coaching as a core method.
- Any injury, diagnosis, or treatment language.

## When blocked
Pick the most reasonable default, log it in DECISIONS.md, continue. Ask only if the choice cannot be undone.
