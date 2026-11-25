"""Test synthetic data schema, determinism, gated-label behavior, and smoke CLI."""

import subprocess
import sys
import time
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PRINCETON_CHECK_TIMEOUT_S, RANDOM_SEED, STANDARDIZED_JOINT_NAMES
from src.m00_data import (
    Sample,
    assign_score_bins,
    check_princeton_labels,
    load_kimore_ex5,
    load_data,
    load_uiprmd,
    make_synthetic_squats,
)


def test_score_bins_use_score_quantiles_not_absolute_probabilities() -> None:
    samples = make_synthetic_squats(75, RANDOM_SEED)
    scored = [replace(sample, clinician_score=float(score), clinician_bin=0)
              for sample, score in zip(samples, np.linspace(0.5, 1.0, 75))]

    binned = assign_score_bins(scored)

    counts = np.bincount([sample.clinician_bin for sample in binned], minlength=3)
    assert counts.min() >= 0.25 * len(binned)
    assert all(original.clinician_bin == 0 for original in scored)


def test_score_bins_rank_fewer_than_three_distinct_scores() -> None:
    samples = make_synthetic_squats(3, RANDOM_SEED)
    scored = [
        replace(samples[0], clinician_score=1.0),
        replace(samples[1], clinician_score=0.0),
        replace(samples[2], clinician_score=None),
    ]

    binned = assign_score_bins(scored)

    assert [sample.clinician_bin for sample in binned] == [1, 0, 0]


def test_synthetic_sample_schema_and_shape() -> None:
    sample = make_synthetic_squats(1, RANDOM_SEED)[0]
    assert isinstance(sample, Sample)
    assert isinstance(sample.subject_id, str)
    assert sample.exercise == "squat"
    assert sample.joints.shape == (6, len(STANDARDIZED_JOINT_NAMES), 3)
    assert sample.joints.dtype == np.float32
    assert sample.clinician_bin in (0, 1, 2)
    assert sample.data_source == "synthetic"
    assert sample.is_synthetic_view is False


def test_synthetic_generation_is_deterministic() -> None:
    first = make_synthetic_squats(3, 17)
    second = make_synthetic_squats(3, 17)
    assert [sample.subject_id for sample in first] == [sample.subject_id for sample in second]
    assert all(np.array_equal(left.joints, right.joints) for left, right in zip(first, second))


def test_princeton_check_is_fast_and_unavailable() -> None:
    started = time.perf_counter()
    assert check_princeton_labels(PRINCETON_CHECK_TIMEOUT_S) is None
    assert time.perf_counter() - started < PRINCETON_CHECK_TIMEOUT_S


def test_smoke_test_cli_is_fast_and_passes() -> None:
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "src.m00_data", "--smoke-test"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "PASS"
    assert time.perf_counter() - started < 5


def test_load_kimore_ex5_reads_real_pickle_layout(tmp_path: Path) -> None:
    positions = np.array(
        [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 8, 9, 10, 11]], dtype=np.float32
    )
    frame = pd.DataFrame(
        {
            "spinebase": [positions, positions + 10],
            "kneelefT": [positions, positions + 20],
            "cTS": [0.8, 0.6],
        }
    )
    path = tmp_path / "KiMoRe" / "kimore_exercise_dataset.pkl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump({"ex5": frame}, handle)
    samples = load_kimore_ex5(str(tmp_path))
    assert len(samples) == 2
    assert len({sample.subject_id for sample in samples}) == 2
    assert all(sample.data_source == "manual" for sample in samples)
    assert all(sample.joints.shape == (2, len(STANDARDIZED_JOINT_NAMES), 3) for sample in samples)
    assert [sample.clinician_score for sample in samples] == [0.8, 0.6]
    assert np.array_equal(samples[0].joints[:, STANDARDIZED_JOINT_NAMES.index("spine_base")], positions[:, 4:7])


def test_load_uiprmd_reads_real_layout_and_groups_subjects(tmp_path: Path) -> None:
    values = ",".join(str(value) for value in range(66))
    for episode in ("01", "02"):
        folder = tmp_path / "UI-PRMD" / "Correct" / "Kinect" / "Positions"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"m01_s01_e{episode}_positions.txt").write_text(values, encoding="utf-8")
    incorrect = tmp_path / "UI-PRMD" / "Incorrect" / "Kinect" / "Positions"
    incorrect.mkdir(parents=True)
    (incorrect / "m01_s02_e01_positions.txt").write_text(values, encoding="utf-8")
    samples = load_uiprmd(str(tmp_path))
    assert len(samples) == 3
    assert {sample.subject_id for sample in samples} == {"uiprmd_s01", "uiprmd_s02"}
    assert [sample.clinician_score for sample in samples].count(1.0) == 2
    assert [sample.clinician_score for sample in samples].count(0.0) == 1
    assert all(sample.data_source == "manual" for sample in samples)
    # Table 2's left lower leg is the 16th triplet, not the third triplet.
    knee = STANDARDIZED_JOINT_NAMES.index("knee_left")
    assert np.array_equal(samples[0].joints[0, knee], [45, 46, 47])


def test_uiprmd_zenodo_archive_layout_and_invalid_width(tmp_path: Path) -> None:
    base = tmp_path / "UI-PRMD"
    correct = base / "Segmented Movements" / "Kinect" / "Positions"
    incorrect = base / "Incorrect Segmented Movements" / "Kinect" / "Positions"
    correct.mkdir(parents=True)
    incorrect.mkdir(parents=True)
    values = ",".join(str(value) for value in range(66))
    (correct / "m01_s01_e01_positions.txt").write_text(values, encoding="utf-8")
    (incorrect / "m01_s01_e02_positions_inc.txt").write_text(values, encoding="utf-8")
    (correct / "m01_s02_e01_positions.txt").write_text(values + ",66", encoding="utf-8")
    samples = load_uiprmd(str(tmp_path))
    assert len(samples) == 2
    assert {sample.subject_id for sample in samples} == {"uiprmd_s01"}
    assert {sample.clinician_score for sample in samples} == {0.0, 1.0}


def test_garbage_data_dir_uses_synthetic_fallback(tmp_path: Path) -> None:
    (tmp_path / "garbage.txt").write_text("not a dataset", encoding="utf-8")
    samples = load_kimore_ex5(str(tmp_path))
    assert len(samples) == 4
    assert all(sample.data_source == "synthetic" for sample in samples)


def test_load_data_falls_back_to_uiprmd_before_synthetic(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "garbage.txt").write_text("not a dataset", encoding="utf-8")
    folder = tmp_path / "UI-PRMD" / "Correct" / "Kinect" / "Positions"
    folder.mkdir(parents=True)
    (folder / "m01_s01_e01_positions.txt").write_text(",".join(str(value) for value in range(66)), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    samples = load_data(str(tmp_path))

    assert len(samples) == 1
    assert all(sample.data_source == "manual" for sample in samples)
    assert json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))["source"] == "manual"
