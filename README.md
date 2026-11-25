HEAD
# form-correction

Squat form-feedback research prototype. See [CLAUDE.md](CLAUDE.md), [DECISIONS.md](DECISIONS.md), [ARCHITECTURE.md](ARCHITECTURE.md).

## Smoke test (local, offline, <5s each)

```powershell
python -m src.m00_data --smoke-test
python -m src.m01_features --smoke-test
python -m src.m02_detector --smoke-test
python -m src.m03_agreement --smoke-test
python -m src.m04_report --smoke-test
python -m app.biocoach_demo --smoke-test
```

## Full run

Open `kaggle_colab_runner.ipynb` on Kaggle or Colab.


