# form-correction

Squat form-feedback research prototype. See [CLAUDE.md](CLAUDE.md), [DECISIONS.md](DECISIONS.md), [ARCHITECTURE.md](ARCHITECTURE.md), [PAPER_DRAFT.md](PAPER_DRAFT.md), and [SLIDES_OUTLINE.md](SLIDES_OUTLINE.md).

## Smoke test (local, offline, <5s each)

```powershell
python -m src.m00_data --smoke-test
python -m src.m01_features --smoke-test
python -m src.m02_detector --smoke-test
python -m src.m03_agreement --smoke-test
python -m src.m04_report --smoke-test
python -m app.biocoach_demo --smoke-test
```

## Full test suite

```powershell
pytest -v
```

74 tests total (65 baseline + 9 for the 2026-11-25 patches).

## Full run

Open `kaggle_colab_runner.ipynb` on Kaggle or Colab. Full chain (cells 1-8) requires Gemini throughput; cells 1-5 + 8 populate the per-flag and detector-vs-clinician diagnostics without touching Gemini.

## Static slide exhibits

```powershell
python scripts/make_slide_exhibits.py
```

Writes three PNGs under `slides_figures/`: a 0° and a 45° stick-figure triptych, plus a composed BioCoach safety exhibit for slide 9 of the deck. Offline, no network.
