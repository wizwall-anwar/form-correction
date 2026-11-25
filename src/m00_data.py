"""Load squat sequences from bounded fallbacks and generate offline test data.

Inputs are an optional manual data directory and configured timeouts. Outputs are
standardized ``Sample`` objects and, for the orchestrator, a small manifest. The
network and gated-data paths are deliberate stubs so this prototype never
downloads data or blocks waiting for an external service.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from .config import (
    CLINICIAN_SCORE_BIN_EDGES,
    KIMORE_SCORE_COLUMN_CANDIDATES,
    KINECT_V2_TO_STANDARD_JOINT_MAP,
    NETWORK_TIMEOUT_S,
    PRINCETON_CHECK_TIMEOUT_S,
    RANDOM_SEED,
    STANDARDIZED_JOINT_NAMES,
    UI_PRMD_LABELS,
    UI_PRMD_SQUAT_MOVEMENT_CODE,
    resolve_data_dir,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class Sample:
    """One standardized skeleton sequence and its optional clinical label."""

    subject_id: str
    session_id: Optional[str]
    seq_id: str
    exercise: str
    joints: np.ndarray
    fps: float
    clinician_score: Optional[float]
    clinician_bin: int
    flags_3d: dict
    view_deg: float
    is_synthetic_view: bool
    data_source: Literal["manual", "fetched", "synthetic"]


def _score_bin(score: Optional[float]) -> int:
    if score is None:
        return 0
    return int(np.digitize(score, CLINICIAN_SCORE_BIN_EDGES, right=False))


def _fetch_stub(timeout_s: float) -> None:
    """Represent a disabled fetch without making a real network call."""

    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")
    raise TimeoutError("network fetch disabled in offline prototype")


def _load_with_fallback(
    data_dir: Optional[str], timeout_s: float, manual_loader: Callable[[Path], list[Sample]]
) -> list[Sample]:
    resolved = resolve_data_dir(data_dir)
    if resolved is not None:
        samples = manual_loader(Path(resolved))
        if samples:
            return samples
    try:
        _fetch_stub(timeout_s)
    except (TimeoutError, OSError):
        return make_synthetic_squats(4, RANDOM_SEED)
    return make_synthetic_squats(4, RANDOM_SEED)


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    normalized = {_normalized(name): name for name in fieldnames}
    for candidate in candidates:
        if _normalized(candidate) in normalized:
            return normalized[_normalized(candidate)]
    return None


def _parse_kimore(root: Path) -> list[Sample]:
    """Parse KIMORE's local Exercise 5 pickle into standardized samples.

    Each row is treated as one unique subject/session because this pickle has
    no subject identifier. Joint cells contain quaternion plus XYZ; only XYZ
    is used. Unknown or malformed rows are logged and skipped.
    """
    path = root / "KiMoRe" / "kimore_exercise_dataset.pkl"
    if not path.is_file():
        return []
    try:
        import pandas as pd

        with path.open("rb") as handle:
            data = pickle.load(handle)
        frame = data.get("ex5") if isinstance(data, dict) else None
        if not isinstance(frame, pd.DataFrame):
            LOGGER.warning("Skipping KIMORE file %s: missing ex5 DataFrame", path)
            return []
        score_column = _find_column([str(column) for column in frame.columns], KIMORE_SCORE_COLUMN_CANDIDATES)
        if score_column is None:
            LOGGER.warning("Skipping KIMORE file %s: no supported score column", path)
            return []
        source_names = {_normalized(name): target for name, target in KINECT_V2_TO_STANDARD_JOINT_MAP.items()}
        joint_columns = {
            column: source_names[_normalized(str(column))]
            for column in frame.columns
            if _normalized(str(column)) in source_names
        }
        if not joint_columns:
            LOGGER.warning("Skipping KIMORE file %s: no recognized joint columns", path)
            return []
        samples: list[Sample] = []
        for index, row in frame.iterrows():
            try:
                score = float(row[score_column])
                arrays = {column: np.asarray(row[column]) for column in joint_columns}
                frame_count = next(len(values) for values in arrays.values() if values.ndim == 2 and values.shape[0])
                if any(values.ndim != 2 or values.shape != (frame_count, 7) for values in arrays.values()):
                    raise ValueError("joint cells are not consistent (T, 7) arrays")
                joints = np.zeros((frame_count, len(STANDARDIZED_JOINT_NAMES), 3), dtype=np.float32)
                for column, values in arrays.items():
                    joint_index = STANDARDIZED_JOINT_NAMES.index(joint_columns[column])
                    joints[:, joint_index] = values[:, 4:7]
            except (StopIteration, TypeError, ValueError, IndexError, KeyError) as error:
                LOGGER.warning("Skipping KIMORE row %s in %s: %s", index, path, error)
                continue
            subject_id = f"kimore_ex5_row_{index}"
            samples.append(Sample(subject_id, subject_id, subject_id, "squat", joints, 30.0, score, _score_bin(score), {}, 0.0, False, "manual"))
        return samples
    except Exception as error:
        LOGGER.warning("Skipping KIMORE file %s: %s", path, error)
        return []


def _parse_uiprmd(root: Path) -> list[Sample]:
    """Parse Kinect UI-PRMD deep-squat position files.

    The exact column-to-joint identity is unverified, so the first joint
    triplets are used positionally; flags should not be trusted like KIMORE's
    named-joint data.
    """
    samples: list[Sample] = []
    pattern = re.compile(r"m(\d{2})_s(\d{2})_e(\d{2})_positions\.txt$", re.IGNORECASE)
    try:
        dataset_root = root / "UI-PRMD"
        for file_path in sorted(dataset_root.rglob("*.txt")):
            relative = [part.lower() for part in file_path.relative_to(dataset_root).parts]
            match = pattern.fullmatch(file_path.name)
            if match is None or match.group(1) != UI_PRMD_SQUAT_MOVEMENT_CODE:
                continue
            if len(relative) != 4 or relative[1] != "kinect" or relative[2] != "positions":
                continue
            label = relative[0]
            if label not in UI_PRMD_LABELS:
                continue
            values = np.genfromtxt(file_path, delimiter=",", dtype=np.float32)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            required_columns = len(STANDARDIZED_JOINT_NAMES) * 3
            if values.ndim != 2 or values.shape[1] < required_columns:
                LOGGER.warning("Skipping UI-PRMD file %s: fewer than %s numeric columns", file_path, required_columns)
                continue
            joints = values[:, :required_columns].reshape(values.shape[0], len(STANDARDIZED_JOINT_NAMES), 3)
            subject_id = f"uiprmd_s{match.group(2)}"
            samples.append(Sample(subject_id, match.group(3), file_path.stem, "squat", joints, 30.0, UI_PRMD_LABELS[label], int(UI_PRMD_LABELS[label]), {}, 0.0, False, "manual"))
    except (OSError, ValueError, IndexError) as error:
        LOGGER.warning("Skipping UI-PRMD data under %s: %s", root, error)
    if samples:
        LOGGER.warning("UI-PRMD joint-column identity is unverified; flags from it should not be trusted like KIMORE named-joint data")
    return samples


def load_kimore_ex5(
    data_dir: Optional[str] = None, timeout_s: float = NETWORK_TIMEOUT_S
) -> list[Sample]:
    """Load Kinect-based KIMORE Exercise 5 squats through the fallback chain.

    KIMORE skeletons come from Kinect; transfer to phone video is untested.
    """

    return _load_with_fallback(data_dir, timeout_s, _parse_kimore)


def load_uiprmd(
    data_dir: Optional[str] = None, timeout_s: float = NETWORK_TIMEOUT_S
) -> list[Sample]:
    """Load UI-PRMD through the same bounded manual, fetch, synthetic chain.

    UI-PRMD's binary correct/incorrect label is represented as score 1.0/0.0;
    its incorrect repetitions were deliberately performed by healthy subjects.
    """

    return _load_with_fallback(data_dir, timeout_s, _parse_uiprmd)


def check_princeton_labels(timeout_s: float = PRINCETON_CHECK_TIMEOUT_S) -> Optional[dict]:
    """Check gated Princeton labels once and return ``None`` when unavailable."""

    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")
    return None


def make_synthetic_squats(n_subjects: int, seed: int) -> list[Sample]:
    """Generate deterministic, clearly synthetic squat sequences offline."""

    if n_subjects < 0:
        raise ValueError("n_subjects must be non-negative")
    rng = np.random.default_rng(seed)
    names = STANDARDIZED_JOINT_NAMES
    index = {name: i for i, name in enumerate(names)}
    samples: list[Sample] = []
    for subject in range(n_subjects):
        depth, lean, asymmetry = (
            rng.uniform(0.35, 0.85),
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.08, 0.08),
        )
        frames = np.linspace(0.0, 1.0, 6)
        joints = np.zeros((len(frames), len(names), 3), dtype=np.float32)
        for frame, phase in enumerate(frames):
            squat = depth * np.sin(np.pi * phase)
            hip_y = 1.0 - 0.35 * squat
            knee_y = 0.55 - 0.22 * squat
            base = {
                "spine_base": (0.0, 0.90 - 0.25 * squat, 0.0),
                "spine_mid": (lean * squat, hip_y + 0.35, 0.0),
                "hip_left": (-0.16, hip_y, 0.0),
                "hip_right": (0.16, hip_y, 0.0),
                "knee_left": (-0.18 - asymmetry * squat, knee_y, 0.10 * squat),
                "knee_right": (0.18 + asymmetry * squat, knee_y, 0.10 * squat),
                "ankle_left": (-0.18, 0.05, 0.0),
                "ankle_right": (0.18, 0.05, 0.0),
                "shoulder_left": (-0.25 + lean * squat, hip_y + 0.70, 0.0),
                "shoulder_right": (0.25 + lean * squat, hip_y + 0.70, 0.0),
            }
            for name, point in base.items():
                joints[frame, index[name]] = point
        score = float(
            np.clip(1.0 - abs(lean) - abs(asymmetry) - max(0.0, 0.65 - depth), 0.0, 1.0)
        )
        samples.append(
            Sample(
                subject_id=f"synthetic_subject_{subject:03d}",
                session_id="synthetic_session",
                seq_id=f"synthetic_seq_{subject:03d}",
                exercise="squat",
                joints=joints,
                fps=30.0,
                clinician_score=score,
                clinician_bin=_score_bin(score),
                flags_3d={},
                view_deg=0.0,
                is_synthetic_view=False,
                data_source="synthetic",
            )
        )
    return samples


def load_data(data_dir: Optional[str] = None) -> list[Sample]:
    """Run manual, disabled-fetch, and synthetic fallbacks and write a manifest."""

    samples = load_kimore_ex5(data_dir=data_dir, timeout_s=NETWORK_TIMEOUT_S)
    if samples and samples[0].data_source == "synthetic":
        uiprmd_samples = load_uiprmd(data_dir=data_dir, timeout_s=NETWORK_TIMEOUT_S)
        if uiprmd_samples and uiprmd_samples[0].data_source != "synthetic":
            samples = uiprmd_samples
    source = samples[0].data_source if samples else "synthetic"
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "manifest.json").write_text(
        json.dumps({"source": source, "sample_count": len(samples), "seed": RANDOM_SEED}, indent=2),
        encoding="utf-8",
    )
    return samples


def _smoke_test() -> bool:
    samples = make_synthetic_squats(n_subjects=4, seed=RANDOM_SEED)
    expected_shape = (6, len(STANDARDIZED_JOINT_NAMES), 3)
    return (
        len(samples) == 4
        and all(isinstance(sample.subject_id, str) for sample in samples)
        and all(
            isinstance(sample.joints, np.ndarray) and sample.joints.shape == expected_shape
            for sample in samples
        )
        and all(sample.data_source == "synthetic" for sample in samples)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print("PASS" if _smoke_test() else "FAIL")
