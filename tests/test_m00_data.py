"""Test synthetic data schema, determinism, gated-label behavior, and smoke CLI."""

import subprocess
import sys
import time

import numpy as np

from src.config import PRINCETON_CHECK_TIMEOUT_S, RANDOM_SEED, STANDARDIZED_JOINT_NAMES
from src.m00_data import Sample, check_princeton_labels, make_synthetic_squats


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
