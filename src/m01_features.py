"""Compute deterministic skeleton views and kinematic measurements.

Inputs are joint arrays shaped [T, J, 3] in the standardized joint order;
2D arrays shaped [T, J, 2] are accepted for measurements. Outputs are
transformed arrays or six scalar/bool measurements used by later modules.
This exists to keep view generation and geometry checks small, reproducible,
and offline. Rotated skeletons are SYNTHETIC views, not real camera views:
they contain no occlusion, pose-estimator error, or clothing effects.
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import (
    KNEE_DEPTH_FLEXION_THRESHOLD_DEG,
    STANDARDIZED_JOINT_NAMES,
    SYMMETRY_KNEE_ANGLE_DIFF_THRESHOLD_DEG,
    TRUNK_LEAN_MAX_ANGLE_DEG,
)


def _as_3d(joints: np.ndarray) -> np.ndarray:
    array = np.asarray(joints)
    if array.ndim != 3 or array.shape[-1] not in (2, 3):
        raise ValueError("joints must have shape [T, J, 2] or [T, J, 3]")
    if array.shape[1] != len(STANDARDIZED_JOINT_NAMES):
        raise ValueError("joints must use the standardized ten-joint order")
    if array.shape[-1] == 2:
        array = np.pad(array, ((0, 0), (0, 0), (0, 1)))
    return array


def rotate_view(joints: np.ndarray, deg: float) -> np.ndarray:
    """Rotate 3D joints about vertical Y; the result is a SYNTHETIC view."""

    if deg == 0:
        return joints
    array = np.asarray(joints)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("joints must have shape [T, J, 3]")
    radians = np.deg2rad(deg)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotated = np.array(array, copy=True)
    rotated[..., 0] = cosine * array[..., 0] + sine * array[..., 2]
    rotated[..., 2] = -sine * array[..., 0] + cosine * array[..., 2]
    return rotated


def add_jitter(joints: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add deterministic zero-mean Gaussian coordinate noise to 3D joints."""

    array = np.asarray(joints)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("joints must have shape [T, J, 3]")
    return array + np.random.default_rng(seed).normal(0.0, sigma, size=array.shape)


def project_2d(joints: np.ndarray) -> np.ndarray:
    """Project 3D joints orthographically by retaining X and Y coordinates."""

    array = np.asarray(joints)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("joints must have shape [T, J, 3]")
    return array[..., :2]


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    denominator = first_norm * second_norm
    cosine = np.divide(
        np.sum(first * second, axis=-1),
        denominator,
        out=np.ones_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def kinematic_flags(joints: np.ndarray) -> dict:
    """Return deterministic knee, trunk, and left-right symmetry measurements."""

    array = _as_3d(joints)
    hip_left, hip_right = array[:, 0], array[:, 1]
    knee_left, knee_right = array[:, 2], array[:, 3]
    ankle_left, ankle_right = array[:, 4], array[:, 5]
    spine_base, spine_mid = array[:, 6], array[:, 7]

    left_knee = 180.0 - _angle_degrees(hip_left - knee_left, ankle_left - knee_left)
    right_knee = 180.0 - _angle_degrees(hip_right - knee_right, ankle_right - knee_right)
    knee_flexion_max = float(np.max(np.maximum(left_knee, right_knee)))
    trunk_lean = _angle_degrees(spine_mid - spine_base, np.array([0.0, 1.0, 0.0]))
    trunk_lean_max = float(np.max(trunk_lean))
    symmetry_diff = float(np.max(np.abs(left_knee - right_knee)))
    return {
        "knee_flexion_max_deg": knee_flexion_max,
        "knee_depth_pass": bool(knee_flexion_max >= KNEE_DEPTH_FLEXION_THRESHOLD_DEG),
        "trunk_lean_max_deg": trunk_lean_max,
        "trunk_lean_pass": bool(trunk_lean_max <= TRUNK_LEAN_MAX_ANGLE_DEG),
        "symmetry_diff_deg": symmetry_diff,
        "symmetry_pass": bool(symmetry_diff <= SYMMETRY_KNEE_ANGLE_DIFF_THRESHOLD_DEG),
    }


def flag_error(flags_view: dict, flags_3d: dict) -> dict:
    """Return absolute view-to-reference differences for the three measurements."""

    keys = ("knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg")
    return {key: float(abs(flags_view[key] - flags_3d[key])) for key in keys}


def _smoke_test() -> bool:
    joints = np.zeros((3, len(STANDARDIZED_JOINT_NAMES), 3), dtype=float)
    joints[:, 0] = (-0.2, 1.0, 0.0)
    joints[:, 1] = (0.2, 1.0, 0.0)
    joints[:, 2] = (-0.2, 0.5, 0.0)
    joints[:, 3] = (0.2, 0.5, 0.0)
    joints[:, 4] = (0.3, 0.5, 0.0)
    joints[:, 5] = (0.7, 0.5, 0.0)
    joints[:, 7] = (0.0, 1.0, 0.0)
    flags = kinematic_flags(joints)
    errors = flag_error(flags, kinematic_flags(rotate_view(joints, 15)))
    expected_flags = {
        "knee_flexion_max_deg",
        "knee_depth_pass",
        "trunk_lean_max_deg",
        "trunk_lean_pass",
        "symmetry_diff_deg",
        "symmetry_pass",
    }
    expected_errors = {"knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg"}
    return (
        rotate_view(joints, 0) is joints
        and np.array_equal(add_jitter(joints, 0.01, 11), add_jitter(joints, 0.01, 11))
        and expected_flags == set(flags)
        and expected_errors == set(errors)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print("PASS" if _smoke_test() else "FAIL")
