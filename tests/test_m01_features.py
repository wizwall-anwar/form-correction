"""Tests for deterministic skeleton feature measurements and view transforms.

Inputs are small hand-written joint arrays; outputs are checked for geometry,
determinism, and the required measurement keys so this module stays offline.
"""

import numpy as np

from src.m01_features import add_jitter, flag_error, kinematic_flags, rotate_view


def _toy_joints() -> np.ndarray:
    joints = np.zeros((1, 10, 3), dtype=float)
    joints[0, 0] = (-0.2, 1.0, 0.0)
    joints[0, 1] = (0.2, 1.0, 0.0)
    joints[0, 2] = (-0.2, 0.5, 0.0)
    joints[0, 3] = (0.2, 0.5, 0.0)
    joints[0, 4] = (0.3, 0.5, 0.0)
    joints[0, 5] = (0.7, 0.5, 0.0)
    joints[0, 6] = (0.0, 0.0, 0.0)
    joints[0, 7] = (0.0, 1.0, 0.0)
    return joints


def test_rotate_zero_degrees_is_identity() -> None:
    joints = _toy_joints()
    assert rotate_view(joints, 0) is joints


def test_add_jitter_is_seeded_and_nonzero() -> None:
    joints = _toy_joints()
    first = add_jitter(joints, sigma=0.1, seed=7)
    second = add_jitter(joints, sigma=0.1, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, joints)


def test_kinematic_flags_have_required_types() -> None:
    flags = kinematic_flags(_toy_joints())
    assert set(flags) == {
        "knee_flexion_min_deg",
        "knee_depth_pass",
        "trunk_lean_max_deg",
        "trunk_lean_pass",
        "symmetry_diff_deg",
        "symmetry_pass",
    }
    assert isinstance(flags["knee_flexion_min_deg"], float)
    assert isinstance(flags["knee_depth_pass"], bool)
    assert isinstance(flags["trunk_lean_max_deg"], float)
    assert isinstance(flags["trunk_lean_pass"], bool)
    assert isinstance(flags["symmetry_diff_deg"], float)
    assert isinstance(flags["symmetry_pass"], bool)


def test_flag_error_has_nonnegative_numeric_measurements() -> None:
    errors = flag_error(
        {"knee_flexion_min_deg": 10, "trunk_lean_max_deg": 20, "symmetry_diff_deg": 3},
        {"knee_flexion_min_deg": 1, "trunk_lean_max_deg": 25, "symmetry_diff_deg": 8},
    )
    assert set(errors) == {
        "knee_flexion_min_deg",
        "trunk_lean_max_deg",
        "symmetry_diff_deg",
    }
    assert errors == {
        "knee_flexion_min_deg": 9.0,
        "trunk_lean_max_deg": 5.0,
        "symmetry_diff_deg": 5.0,
    }
    assert all(isinstance(value, float) and value >= 0 for value in errors.values())


def test_toy_squat_has_hand_computed_ninety_degree_knee_flexion() -> None:
    flags = kinematic_flags(_toy_joints())
    assert np.isclose(flags["knee_flexion_min_deg"], 90.0, atol=1e-6)

