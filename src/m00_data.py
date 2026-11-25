"""Load squat sequences from bounded fallbacks and generate offline test data.

Inputs are an optional manual data directory and configured timeouts. Outputs are
standardized ``Sample`` objects and, for the orchestrator, a small manifest. The
network and gated-data paths are deliberate stubs so this prototype never
downloads data or blocks waiting for an external service.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .config import (
    CLINICIAN_SCORE_BIN_EDGES,
    NETWORK_TIMEOUT_S,
    PRINCETON_CHECK_TIMEOUT_S,
    RANDOM_SEED,
    STANDARDIZED_JOINT_NAMES,
)


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


def _load_with_fallback(data_dir: Optional[str], timeout_s: float) -> list[Sample]:
    if data_dir is not None and Path(data_dir).exists():
        # No input format is configured yet; continue to the bounded fetch stub.
        pass
    try:
        _fetch_stub(timeout_s)
    except (TimeoutError, OSError):
        return make_synthetic_squats(4, RANDOM_SEED)
    return make_synthetic_squats(4, RANDOM_SEED)


def load_kimore_ex5(
    data_dir: Optional[str] = None, timeout_s: float = NETWORK_TIMEOUT_S
) -> list[Sample]:
    """Load Kinect-based KIMORE Exercise 5 squats through the fallback chain.

    KIMORE skeletons come from Kinect; transfer to phone video is untested. The
    manual and network paths are placeholders and never download real datasets.
    """

    return _load_with_fallback(data_dir, timeout_s)


def load_uiprmd(
    data_dir: Optional[str] = None, timeout_s: float = NETWORK_TIMEOUT_S
) -> list[Sample]:
    """Load UI-PRMD through the same bounded manual, fetch, synthetic chain.

    UI-PRMD's binary correct/incorrect label is represented as clinician score
    1.0/0.0 by a future real loader; this offline placeholder returns synthetic
    squat samples.
    """

    return _load_with_fallback(data_dir, timeout_s)


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
