"""Central constants for the form-analysis pipeline.

Inputs are skeleton views, clinician scores, and optional network-backed model
requests; outputs are shared seeds, thresholds, binning rules, timeouts, and
joint-name mappings. Keeping them here prevents modules from drifting apart
and makes experiments reproducible.
"""

import os
from pathlib import Path
from typing import Final, Optional


RANDOM_SEED: Final[int] = 20261124
ROTATION_ANGLES_DEG: Final[tuple[int, ...]] = (0, 15, 30, 45, 60)
JITTER_SIGMA: Final[float] = 0.01
NETWORK_TIMEOUT_S: Final[int] = 30
PRINCETON_CHECK_TIMEOUT_S: Final[int] = 15
FORM_CORRECTION_DATA_DIR_ENV: Final[str] = "FORM_CORRECTION_DATA_DIR"
KIMORE_EXERCISE_DIR_NAME: Final[str] = "Es5"
KIMORE_SCORE_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "ClinicalScore",
    "TotalScore",
    "Score",
    "cTS",
)
UI_PRMD_SQUAT_MOVEMENT_CODE: Final[str] = "01"
UI_PRMD_LABELS: Final[dict[str, float]] = {"correct": 1.0, "incorrect": 0.0}

# Tertile cuts are quantiles of clinician scores from TRAIN subjects only.
CLINICIAN_SCORE_BIN_QUANTILES: Final[tuple[float, float]] = (1 / 3, 2 / 3)
RIDGE_ALPHA: Final[float] = 1.0
BOOTSTRAP_RESAMPLES: Final[int] = 1000
SELECTIVE_COVERAGES: Final[tuple[float, ...]] = (0.5, 0.8, 1.0)

# Fit and validate these deterministic flag thresholds on train-split subjects only, never test subjects.
KNEE_DEPTH_FLEXION_THRESHOLD_DEG: Final[float] = 90.0
TRUNK_LEAN_MAX_ANGLE_DEG: Final[float] = 20.0
SYMMETRY_KNEE_ANGLE_DIFF_THRESHOLD_DEG: Final[float] = 15.0

ABSTAIN_CONFIDENCE_THRESHOLD: Final[float] = 0.70
VLM_BACKEND: Final[str] = "placeholder"
VLM_TIMEOUT_S: Final[int] = 30
GEMINI_MODEL_ID: Final[str] = "gemini-3.1-flash-lite"
GEMINI_MODEL_FALLBACKS: Final[tuple[str, ...]] = (
    GEMINI_MODEL_ID,
    "gemini-2.5-flash-lite",
)
assert GEMINI_MODEL_FALLBACKS[0] == GEMINI_MODEL_ID
VLM_CACHE_DIR: Final[str] = "results/vlm_cache"
VLM_RATE_LIMIT_DELAY_S: Final[float] = 4.0
VLM_RETRY_MAX_ATTEMPTS: Final[int] = 4
VLM_RETRY_BASE_DELAY_S: Final[float] = 1.0
VLM_RETRY_MAX_DELAY_S: Final[float] = 30.0
VLM_RETRY_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
VLM_ERROR_BODY_PREVIEW_CHARS: Final[int] = 200

STANDARDIZED_JOINT_NAMES: Final[tuple[str, ...]] = (
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
    "spine_base",
    "spine_mid",
    "shoulder_left",
    "shoulder_right",
)

# Kinect v2's 25 names; distal points without a standardized slot collapse to the nearest body landmark.
KINECT_V2_TO_STANDARD_JOINT_MAP: Final[dict[str, str]] = {
    "SpineBase": "spine_base",
    "SpineMid": "spine_mid",
    "Neck": "spine_mid",
    "Head": "spine_mid",
    "ShoulderLeft": "shoulder_left",
    "ElbowLeft": "shoulder_left",
    "WristLeft": "shoulder_left",
    "HandLeft": "shoulder_left",
    "ShoulderRight": "shoulder_right",
    "ElbowRight": "shoulder_right",
    "WristRight": "shoulder_right",
    "HandRight": "shoulder_right",
    "HipLeft": "hip_left",
    "KneeLeft": "knee_left",
    "AnkleLeft": "ankle_left",
    "FootLeft": "ankle_left",
    "HipRight": "hip_right",
    "KneeRight": "knee_right",
    "AnkleRight": "ankle_right",
    "FootRight": "ankle_right",
    "SpineShoulder": "spine_mid",
    "HandTipLeft": "shoulder_left",
    "ThumbLeft": "shoulder_left",
    "HandTipRight": "shoulder_right",
    "ThumbRight": "shoulder_right",
}


def _find_dataset_root(base: Path, max_depth: int = 5) -> Optional[str]:
    """Return the first nearby directory containing a supported dataset."""

    if not base.is_dir():
        return None
    for root, directories, _ in os.walk(base):
        directories.sort()
        root_path = Path(root)
        depth = len(root_path.relative_to(base).parts)
        if any((root_path / name).is_dir() for name in ("KiMoRe", "UI-PRMD")):
            return str(root_path)
        if depth >= max_depth:
            directories.clear()
    return None


def resolve_data_dir(data_dir: Optional[str] = None) -> Optional[str]:
    """Resolve an existing local dataset directory without touching the network.

    Inputs are an optional explicit path and the ``FORM_CORRECTION_DATA_DIR``
    environment variable. Outputs are the first directory containing a supported
    dataset under those paths, common Colab paths, and Kaggle dataset mounts, or
    ``None``. Discovery descends at most five levels and never downloads or
    modifies any filesystem.
    """

    candidates = [data_dir, os.environ.get(FORM_CORRECTION_DATA_DIR_ENV)]
    candidates.extend(
        (
            "/content/drive/MyDrive/form-correction-data",
            "/content/data",
            "/kaggle/input",
        )
    )
    for candidate in candidates:
        if candidate:
            dataset_root = _find_dataset_root(Path(candidate))
            if dataset_root is not None:
                return dataset_root
    return None
