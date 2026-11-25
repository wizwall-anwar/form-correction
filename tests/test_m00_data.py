"""Test synthetic data schema, determinism, gated-label behavior, and smoke CLI."""

import subprocess
import sys
import time
import json
from pathlib import Path

import numpy as np

from src.config import PRINCETON_CHECK_TIMEOUT_S, RANDOM_SEED, STANDARDIZED_JOINT_NAMES
from src.m00_data import (
    Sample,
    check_princeton_labels,
    load_kimore_ex5,
    load_data,
    load_uiprmd,
    make_synthetic_squats,
)


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


def test_load_kimore_ex5_reads_manual_layout(tmp_path: Path) -> None:
    for subject in ("S01", "S02"):
        root = tmp_path / "KiMoRe" / "GPP" / subject / "Es5"
        (root / "Raw").mkdir(parents=True)
        (root / "Label").mkdir()
        (root / "Raw" / "squat.csv").write_text(
            "SpineBase_X,SpineBase_Y,SpineBase_Z,KneeLeft_X,KneeLeft_Y,KneeLeft_Z,"
            "KneeRight_X,KneeRight_Y,KneeRight_Z\n"
            "0,1,0,-.2,.5,0,.2,.5,0\n"
            "0,.9,0,-.2,.4,0,.2,.4,0\n",
            encoding="utf-8",
        )
        (root / "Label" / "score.csv").write_text(
            "ClinicalScore\n0.8\n", encoding="utf-8"
        )

    samples = load_kimore_ex5(str(tmp_path))
    assert len(samples) == 2
    assert {sample.subject_id for sample in samples} == {"GPP/S01", "GPP/S02"}
    assert all(sample.data_source == "manual" for sample in samples)
    assert all(sample.joints.shape == (2, len(STANDARDIZED_JOINT_NAMES), 3) for sample in samples)
    assert all(sample.clinician_score == 0.8 for sample in samples)


def test_load_uiprmd_reads_manual_layout(tmp_path: Path) -> None:
    for subject, label in (("Subject1", "Correct"), ("Subject2", "Incorrect")):
        folder = tmp_path / "UI-PRMD" / subject / "Positions" / label
        folder.mkdir(parents=True)
        rows = [" ".join(str(value) for value in range(30)), " ".join(str(value + 1) for value in range(30))]
        (folder / "DeepSquat_1.txt").write_text("\n".join(rows), encoding="utf-8")

    samples = load_uiprmd(str(tmp_path))
    assert len(samples) == 2
    assert {sample.subject_id for sample in samples} == {"Subject1", "Subject2"}
    assert [sample.clinician_score for sample in samples] == [1.0, 0.0]
    assert all(sample.data_source == "manual" for sample in samples)


def test_garbage_data_dir_uses_synthetic_fallback(tmp_path: Path) -> None:
    (tmp_path / "garbage.txt").write_text("not a dataset", encoding="utf-8")
    samples = load_kimore_ex5(str(tmp_path))
    assert len(samples) == 4
    assert all(sample.data_source == "synthetic" for sample in samples)


def test_load_data_falls_back_to_uiprmd_before_synthetic(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "garbage.txt").write_text("not a dataset", encoding="utf-8")
    folder = tmp_path / "UI-PRMD" / "Subject1" / "Positions" / "Correct"
    folder.mkdir(parents=True)
    (folder / "DeepSquat_1.txt").write_text(
        "\n".join(" ".join(str(value + offset) for value in range(30)) for offset in (0, 1)),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    samples = load_data(str(tmp_path))

    assert len(samples) == 1
    assert all(sample.data_source == "manual" for sample in samples)
    assert json.loads((tmp_path / "results" / "manifest.json").read_text(encoding="utf-8"))["source"] == "manual"
