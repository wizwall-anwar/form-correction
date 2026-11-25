"""Verify the shared pipeline configuration is complete and typed as expected."""

from pathlib import Path

from src.config import (
    KINECT_V2_TO_STANDARD_JOINT_MAP,
    KNEE_DEPTH_FLEXION_THRESHOLD_DEG,
    RANDOM_SEED,
    ROTATION_ANGLES_DEG,
    STANDARDIZED_JOINT_NAMES,
    SYMMETRY_KNEE_ANGLE_DIFF_THRESHOLD_DEG,
    TRUNK_LEAN_MAX_ANGLE_DEG,
    _find_dataset_root,
)


def test_config_constants_cover_required_values() -> None:
    assert set(STANDARDIZED_JOINT_NAMES) <= set(KINECT_V2_TO_STANDARD_JOINT_MAP.values())
    assert 0 in ROTATION_ANGLES_DEG
    assert isinstance(RANDOM_SEED, int)
    assert all(
        isinstance(value, (int, float))
        for value in (
            KNEE_DEPTH_FLEXION_THRESHOLD_DEG,
            TRUNK_LEAN_MAX_ANGLE_DEG,
            SYMMETRY_KNEE_ANGLE_DIFF_THRESHOLD_DEG,
        )
    )


def test_find_dataset_root_handles_data_wrapper(tmp_path: Path) -> None:
    expected = tmp_path / "data"
    (expected / "KiMoRe").mkdir(parents=True)

    assert _find_dataset_root(tmp_path) == str(expected)


def test_find_dataset_root_handles_deep_kaggle_layout(tmp_path: Path) -> None:
    expected = tmp_path / "datasets" / "user" / "dataset-name" / "data"
    (expected / "KiMoRe").mkdir(parents=True)

    assert _find_dataset_root(tmp_path) == str(expected)


def test_find_dataset_root_stops_at_max_depth(tmp_path: Path) -> None:
    too_deep = tmp_path / "one" / "two" / "three"
    (too_deep / "KiMoRe").mkdir(parents=True)

    assert _find_dataset_root(tmp_path, max_depth=2) is None


def test_find_dataset_root_accepts_base_itself(tmp_path: Path) -> None:
    (tmp_path / "UI-PRMD").mkdir()

    assert _find_dataset_root(tmp_path) == str(tmp_path)
