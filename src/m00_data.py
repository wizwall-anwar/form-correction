"""Load squat sequences from bounded fallbacks and generate offline test data.

Inputs are an optional manual data directory and configured timeouts. Outputs are
standardized ``Sample`` objects and, for the orchestrator, a small manifest. The
network and gated-data paths are deliberate stubs so this prototype never
downloads data or blocks waiting for an external service.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from .config import (
    CLINICIAN_SCORE_BIN_EDGES,
    KIMORE_EXERCISE_DIR_NAME,
    KIMORE_SCORE_COLUMN_CANDIDATES,
    KINECT_V2_TO_STANDARD_JOINT_MAP,
    NETWORK_TIMEOUT_S,
    PRINCETON_CHECK_TIMEOUT_S,
    RANDOM_SEED,
    STANDARDIZED_JOINT_NAMES,
    UI_PRMD_LABELS,
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    normalized = {_normalized(name): name for name in fieldnames}
    for candidate in candidates:
        if _normalized(candidate) in normalized:
            return normalized[_normalized(candidate)]
    return None


def _joint_columns(fieldnames: list[str]) -> dict[str, tuple[str, str, str]]:
    columns: dict[str, dict[str, str]] = {}
    source_names = set(KINECT_V2_TO_STANDARD_JOINT_MAP) | set(STANDARDIZED_JOINT_NAMES)
    for field in fieldnames:
        normalized = _normalized(field)
        for source_name in source_names:
            source = _normalized(source_name)
            for axis in ("x", "y", "z"):
                if normalized in (source + axis, axis + source):
                    target = KINECT_V2_TO_STANDARD_JOINT_MAP.get(source_name, source_name)
                    columns.setdefault(target, {})[axis] = field
    return {
        target: (axes["x"], axes["y"], axes["z"])
        for target, axes in columns.items()
        if {"x", "y", "z"}.issubset(axes)
    }


def _parse_kimore(root: Path) -> list[Sample]:
    """Parse ``Es5/Raw/*.csv`` and ``Es5/Label/*.csv`` KIMORE folders.

    The coded layout assumption is ``<group>/<subject>/Es5/Raw`` for a CSV
    with per-frame ``Joint_X/Y/Z`` columns and a sibling ``Label`` CSV with a
    ``ClinicalScore``, ``TotalScore``, or ``Score`` column. Missing or changed
    schemas are logged and skipped per subject, never raised to the pipeline.
    """

    samples: list[Sample] = []
    exercise_dirs = [
        path for path in root.rglob("*") if path.is_dir() and path.name.lower() == KIMORE_EXERCISE_DIR_NAME.lower()
    ]
    for exercise_dir in sorted(exercise_dirs):
        raw_dir = next((p for p in exercise_dir.iterdir() if p.is_dir() and p.name.lower() == "raw"), None)
        label_dir = next((p for p in exercise_dir.iterdir() if p.is_dir() and p.name.lower() == "label"), None)
        raw_files = sorted(raw_dir.glob("*.csv")) if raw_dir else []
        label_files = sorted(label_dir.glob("*.csv")) if label_dir else []
        if not raw_files or not label_files:
            LOGGER.warning("Skipping KIMORE subject at %s: missing Raw or Label CSV", exercise_dir)
            continue
        label_rows = []
        for label_file in label_files:
            label_rows.extend(_read_csv(label_file))
        score_column = _find_column(
            list(label_rows[0]) if label_rows else [], KIMORE_SCORE_COLUMN_CANDIDATES
        )
        if score_column is None or not label_rows:
            LOGGER.warning("Skipping KIMORE subject at %s: no supported score column", exercise_dir)
            continue
        try:
            score = float(label_rows[0][score_column])
        except (TypeError, ValueError):
            LOGGER.warning("Skipping KIMORE subject at %s: score column %s is not numeric", exercise_dir, score_column)
            continue
        raw_file = raw_files[0]
        rows = _read_csv(raw_file)
        columns = _joint_columns(list(rows[0]) if rows else [])
        if not columns:
            LOGGER.warning("Skipping KIMORE file %s: no recognized joint X/Y/Z columns", raw_file)
            continue
        joints = np.zeros((len(rows), len(STANDARDIZED_JOINT_NAMES), 3), dtype=np.float32)
        for joint_name, (x_col, y_col, z_col) in columns.items():
            joint_index = STANDARDIZED_JOINT_NAMES.index(joint_name)
            for frame, row in enumerate(rows):
                try:
                    joints[frame, joint_index] = [float(row[x_col]), float(row[y_col]), float(row[z_col])]
                except (KeyError, TypeError, ValueError):
                    joints = np.empty((0, len(STANDARDIZED_JOINT_NAMES), 3), dtype=np.float32)
                    break
            if not len(joints):
                break
        if not len(joints):
            LOGGER.warning("Skipping KIMORE file %s: non-numeric joint value", raw_file)
            continue
        relative = exercise_dir.relative_to(root).parts
        subject_id = "/".join(relative[-3:-1]) if len(relative) >= 3 else exercise_dir.parent.name
        samples.append(Sample(subject_id, exercise_dir.parent.name, raw_file.stem, "squat", joints, 30.0, score, _score_bin(score), {}, 0.0, False, "manual"))
    return samples


def _parse_uiprmd(root: Path) -> list[Sample]:
    """Parse numeric UI-PRMD deep-squat files from a local extracted folder.

    The coded layout assumption is ``<subject>/(Positions|Angles)/
    (Correct|Incorrect)/*squat*.txt``. Each file is whitespace-delimited (or
    comma-delimited) with at least 30 numeric columns representing 10 joints
    times XYZ; the first 30 columns are standardized in file order. This
    accepts the common Vicon/Kinect subsets and preserves deliberate
    UI-PRMD ``Incorrect`` labels as dataset labels, not real-world judgments.
    """

    samples: list[Sample] = []
    for file_path in sorted(root.rglob("*.txt")):
        parts = [part.lower() for part in file_path.parts]
        if not any("squat" in part.replace("_", "") for part in parts):
            continue
        label_part = next((part for part in parts if part in UI_PRMD_LABELS), None)
        subset_index = next((i for i, part in enumerate(parts) if part in ("positions", "angles")), None)
        if label_part is None or subset_index is None or subset_index == 0:
            continue
        try:
            values = np.genfromtxt(file_path, dtype=np.float32)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[1] < len(STANDARDIZED_JOINT_NAMES) * 3:
                raise ValueError("fewer than 30 numeric columns")
        except (OSError, ValueError):
            LOGGER.warning("Skipping UI-PRMD file %s: unsupported numeric layout", file_path)
            continue
        joints = values[:, : len(STANDARDIZED_JOINT_NAMES) * 3].reshape(
            values.shape[0], len(STANDARDIZED_JOINT_NAMES), 3
        )
        samples.append(Sample(file_path.parts[subset_index - 1], file_path.parent.name, file_path.stem, "squat", joints, 30.0, UI_PRMD_LABELS[label_part], int(UI_PRMD_LABELS[label_part]), {}, 0.0, False, "manual"))
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
