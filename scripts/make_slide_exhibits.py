"""Render static slide exhibits for the paper and deck without a live run.

Inputs are a representative detector record (defaults to the first row of
``results_bundle/halluc_log.jsonl``, which carries real flag numbers from the
clean 2026-11-25 KIMORE Exercise 5 run) plus an optional path to the KIMORE
pickle for the stick-figure triptych. Outputs are three PNGs under
``slides_figures/``:

* ``fig_stick_triptych_0deg.png`` and ``fig_stick_triptych_45deg.png`` — the
  start/bottom/end keyframes of the same squat at two synthetic view angles.
* ``fig_biocoach_safety_exhibit.png`` — the 45-degree triptych paired with
  the deterministic BioCoach bullets and a red-X free-form fabrication the
  verifier rejects, so slide 8 has a static "failure case plus grounded
  output" exhibit without a live demo.

Rationale: slide delivery stays reliable even when Gemini is flaky.
Mirrors the DECISIONS section 5 (2026-11-25) offline-inspection pattern:
plain Python, no pipeline run, no network, no training.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.biocoach_demo import run_demo
from src.config import KINECT_V2_TO_STANDARD_JOINT_MAP, STANDARDIZED_JOINT_NAMES
from src.m00_data import make_synthetic_squats, RANDOM_SEED
from src.m01_features import kinematic_flags, rotate_view
from src.m03_agreement import render_keyframes, render_keyframes_image


_PLANTED_FABRICATION = (
    "The knee flexion is 999.0 degrees and trunk lean is 888.0 degrees."
)

# The first real halluc_log record from the 2026-11-25 clean Colab run.
# Baked in so this script works even without the bundle on hand; override
# with --halluc-log if a fresher one is available.
_DEFAULT_DETECTOR_RECORD = {
    "flags": {
        "knee_flexion_max_deg": 106.72062519573414,
        "knee_depth_pass": True,
        "trunk_lean_max_deg": 26.853276762147964,
        "trunk_lean_pass": False,
        "symmetry_diff_deg": 19.593467990745737,
        "symmetry_pass": False,
    },
    "view_deg": 0.0,
    "pred_bin": 1,
    "confidence": 0.9983445247076589,
    "abstain_flag": False,
}


def _load_detector_record(halluc_log: Optional[Path]) -> dict:
    if halluc_log is None or not halluc_log.is_file():
        return _DEFAULT_DETECTOR_RECORD
    with halluc_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            logged = json.loads(line)
            if "detector_out" in logged:
                return logged["detector_out"]
    return _DEFAULT_DETECTOR_RECORD


def _load_representative_joints(pickle_path: Optional[Path]) -> np.ndarray:
    """Load KIMORE row 0's joints when available, else build a synthetic squat.

    Follows DECISIONS section 5 (2026-11-25): plain-Python pickle read, not
    a pipeline run, used only to produce a visual exhibit. Falls back to the
    synthetic generator so the script still ships an exhibit when the pickle
    is absent (e.g., when running in CI or a clean clone).
    """

    if pickle_path is None or not pickle_path.is_file():
        samples = make_synthetic_squats(4, RANDOM_SEED)
        return samples[2].joints
    try:
        import pandas as pd  # noqa: F401 — needed to unpickle the KIMORE DataFrame

        with pickle_path.open("rb") as handle:
            data = pickle.load(handle)
    except Exception:
        samples = make_synthetic_squats(4, RANDOM_SEED)
        return samples[2].joints
    frame = data.get("ex5") if isinstance(data, dict) else None
    if frame is None:
        samples = make_synthetic_squats(4, RANDOM_SEED)
        return samples[2].joints
    row = frame.iloc[0]
    arrays: dict[str, np.ndarray] = {}
    source_to_target = {name.lower(): target for name, target in KINECT_V2_TO_STANDARD_JOINT_MAP.items()}
    for column in frame.columns:
        key = str(column).lower()
        if key in source_to_target:
            arrays[source_to_target[key]] = np.asarray(row[column])
    reference = next(iter(arrays.values()))
    joints = np.zeros((reference.shape[0], len(STANDARDIZED_JOINT_NAMES), 3), dtype=np.float32)
    for index, name in enumerate(STANDARDIZED_JOINT_NAMES):
        if name in arrays:
            joints[:, index] = arrays[name][:, 4:7]
    return joints


def _write_triptych(joints: np.ndarray, angle: float, out: Path) -> None:
    """Save the start/bottom/end keyframes of ``joints`` rotated to ``angle``."""

    rotated = rotate_view(joints, float(angle))
    keyframes = render_keyframes(rotated)
    png_bytes = render_keyframes_image(keyframes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)


def _write_safety_exhibit(joints: np.ndarray, detector_record: dict,
                          out: Path) -> None:
    """Pair the 45-degree triptych with grounded bullets and a rejected string."""

    rotated = rotate_view(joints, 45.0)
    keyframes = render_keyframes(rotated)
    bullets = run_demo(detector_record)

    figure = plt.figure(figsize=(12, 5.5))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    left = figure.add_subplot(grid[0, 0])
    stick_png = io.BytesIO(render_keyframes_image(keyframes))
    left.imshow(plt.imread(stick_png))
    left.axis("off")
    left.set_title("Stick-figure keyframes at 45 degrees (SYNTHETIC view)",
                   fontsize=10)

    right = figure.add_subplot(grid[0, 1])
    right.axis("off")
    lines: list[tuple[str, str]] = [
        ("[rejected]", f'Free-form: "{_PLANTED_FABRICATION}"'),
        ("", ""),
        ("[grounded]", bullets[0]),
        ("[grounded]", bullets[1]),
        ("[grounded]", bullets[2]),
    ]
    y = 0.95
    for marker, text in lines:
        color = "#b00020" if marker == "[rejected]" else (
            "#206a1b" if marker == "[grounded]" else "black"
        )
        right.text(0.0, y, marker, fontsize=11, color=color,
                   fontweight="bold", transform=right.transAxes)
        right.text(0.16, y, text, fontsize=10, color="#222",
                   transform=right.transAxes, wrap=True, va="top")
        y -= 0.16
    right.text(0.0, 0.03,
               "verifier rejects numbers absent from detector output; "
               "banned terms rejected; only measured values reach the bullets",
               fontsize=8, color="#555", transform=right.transAxes,
               style="italic")
    figure.suptitle(
        "BioCoach safety exhibit: 45 deg view, grounded bullets vs rejected free-form",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", type=Path, default=Path("data/KiMoRe/kimore_exercise_dataset.pkl"),
                        help="KIMORE pickle path; falls back to synthetic if absent")
    parser.add_argument("--halluc-log", type=Path, default=Path("results/halluc_log.jsonl"),
                        help="halluc_log.jsonl from the last bundle; "
                             "defaults to baked-in 2026-11-25 record if absent")
    parser.add_argument("--out-dir", type=Path, default=Path("slides_figures"),
                        help="where to write the exhibit PNGs")
    args = parser.parse_args()

    joints = _load_representative_joints(args.pickle)
    detector_record = _load_detector_record(args.halluc_log)

    _write_triptych(joints, 0.0, args.out_dir / "fig_stick_triptych_0deg.png")
    _write_triptych(joints, 45.0, args.out_dir / "fig_stick_triptych_45deg.png")
    _write_safety_exhibit(joints, detector_record, args.out_dir / "fig_biocoach_safety_exhibit.png")
    print(f"wrote: {args.out_dir}/fig_stick_triptych_0deg.png")
    print(f"wrote: {args.out_dir}/fig_stick_triptych_45deg.png")
    print(f"wrote: {args.out_dir}/fig_biocoach_safety_exhibit.png")


if __name__ == "__main__":
    main()
