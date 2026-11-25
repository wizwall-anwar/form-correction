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
    _find_kaggle_dataset_dir,
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


def test_find_kaggle_dataset_dir_scans_immediate_mounts(tmp_path: Path) -> None:
    (tmp_path / "ignored" / "nested" / "KiMoRe").mkdir(parents=True)
    expected = tmp_path / "user-chosen-slug"
    (expected / "UI-PRMD").mkdir(parents=True)

    assert _find_kaggle_dataset_dir(tmp_path) == str(expected)
