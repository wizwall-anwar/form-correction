"""Build four slide figures from the latest Colab bundle and the local KIMORE pkl.

Reads results_bundle_dir (default: the Windows Downloads bundle folder) plus the
local KIMORE pickle, writes four PNGs under slides_figures/ for the MIS 599
creative component slides. Offline, no training, no network. One-off diagnostic
script, not part of the pipeline.

Run from the repo root:
    python scripts/make_slide_figures.py
    python scripts/make_slide_figures.py --bundle "C:/path/to/results_bundle"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m00_data import load_kimore_ex5
from src.m01_features import kinematic_flags
from src.m02_detector import spearman_rho


DEFAULT_BUNDLE = Path(r"C:/Users/wizwa/Downloads/results_bundle (4)")
OUT = Path("slides_figures")
FLAG_KEYS = ("knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg")
FLAG_LABELS = ("Knee flexion max", "Trunk lean max", "Left-right symmetry diff")


def fig_per_flag_scatter() -> None:
    """Scatter each kinematic flag against the clinician score."""

    samples = [s for s in load_kimore_ex5() if s.clinician_score is not None]
    scores = np.array([float(s.clinician_score) for s in samples])
    flag_rows = [kinematic_flags(s.joints) for s in samples]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, key, label in zip(axes, FLAG_KEYS, FLAG_LABELS):
        values = np.array([row[key] for row in flag_rows])
        rho = spearman_rho(values.tolist(), scores.tolist())
        axis.scatter(values, scores, s=22, alpha=0.75, edgecolor="k", linewidth=0.3)
        axis.set_xlabel(f"{label} (deg)")
        axis.set_ylabel("Clinician score (cTS, higher = better)")
        axis.set_title(f"{label}\nSpearman rho = {rho:+.2f}")
        axis.grid(True, alpha=0.3)
    fig.suptitle(
        f"Per-flag vs clinician score (n = {len(samples)} KIMORE Ex5 subjects)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_flag_vs_score.png", dpi=150)
    plt.close(fig)


def fig_multi_view(bundle: Path) -> None:
    """Angle vs rho (left axis) and mean flag error (right axis) from table2."""

    rows = json.loads((bundle / "table2.json").read_text())["rows"]
    angles = [row["angle"] for row in rows]
    rhos = [row["spearman_rho"] for row in rows]
    errors = [row["flag_error"] for row in rows]
    fig, left = plt.subplots(figsize=(7.5, 4.5))
    left.plot(angles, rhos, "o-", color="tab:red", label="Spearman rho (model vs clinician)")
    left.set_xlabel("View rotation (degrees, SYNTHETIC)")
    left.set_ylabel("Spearman rho", color="tab:red")
    left.tick_params(axis="y", labelcolor="tab:red")
    left.axhline(0, linestyle=":", color="grey", alpha=0.6)
    left.grid(True, alpha=0.3)
    right = left.twinx()
    right.plot(angles, errors, "s--", color="tab:blue", label="Mean flag error (deg)")
    right.set_ylabel("Mean flag error (deg)", color="tab:blue")
    right.tick_params(axis="y", labelcolor="tab:blue")
    left.set_title(
        "Multi-view degradation: flag measurements are rotation-stable (~2 deg),\n"
        "but the baseline's negative rho persists at every angle"
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_multi_view_degradation.png", dpi=150)
    plt.close(fig)


def fig_vlm_distribution(bundle: Path) -> None:
    """Gemini vs clinician bin distribution and confusion matrix at 0 and 45 deg."""

    kappa = json.loads((bundle / "table4.json").read_text())["kappa_report"][
        "model_vs_clinician_reliability"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for column, deg in enumerate(("0", "45")):
        metrics = kappa[deg]
        vlm = np.asarray(metrics["vlm_bins"], dtype=int)
        clin = np.asarray(metrics["clinician_bins"], dtype=int)
        bars = axes[0, column]
        edges = np.arange(4) - 0.5
        bars.hist(vlm, bins=edges, alpha=0.7, label="Gemini", color="tab:orange", edgecolor="k")
        bars.hist(clin, bins=edges, alpha=0.5, label="Clinician", color="tab:blue", edgecolor="k")
        bars.set_xticks([0, 1, 2])
        bars.set_xlabel("Bin")
        bars.set_ylabel("Count")
        bars.set_title(
            f"{deg} deg: n_scored = {len(vlm)}, Cohen kappa = {metrics['cohen_kappa']:+.2f}"
        )
        bars.legend()
        bars.grid(True, alpha=0.3, axis="y")
        matrix = axes[1, column]
        confusion = np.zeros((3, 3), dtype=int)
        for clinician_bin, vlm_bin in zip(clin, vlm):
            confusion[int(clinician_bin), int(vlm_bin)] += 1
        image = matrix.imshow(confusion, cmap="Blues")
        matrix.set_xticks([0, 1, 2])
        matrix.set_yticks([0, 1, 2])
        matrix.set_xlabel("Gemini bin")
        matrix.set_ylabel("Clinician bin")
        threshold = confusion.max() / 2 if confusion.max() else 1
        for i in range(3):
            for j in range(3):
                color = "white" if confusion[i, j] > threshold else "black"
                matrix.text(j, i, confusion[i, j], ha="center", va="center", color=color)
        matrix.set_title("Confusion matrix")
        fig.colorbar(image, ax=matrix, fraction=0.046, pad=0.04)
    fig.suptitle("Gemini collapses almost every sample to bin 0 or 2 and avoids bin 1")
    fig.tight_layout()
    fig.savefig(OUT / "fig_vlm_distribution.png", dpi=150)
    plt.close(fig)


def fig_grounded_vs_free_form(bundle: Path) -> None:
    """Grounded bullets vs planted free-form fabrication from halluc_log.jsonl."""

    logs = [
        json.loads(line)
        for line in (bundle / "halluc_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    free = [item["free_form_unsupported_count"] for item in logs]
    grounded = [item["grounded_unsupported_count"] for item in logs]
    fig, (bars, text) = plt.subplots(1, 2, figsize=(12, 4.8))
    xs = np.arange(len(logs))
    width = 0.35
    bars.bar(xs - width / 2, free, width, label="Free-form (planted 999/888)", color="tab:red")
    bars.bar(
        xs + width / 2,
        grounded,
        width,
        label="Grounded (BioCoach + verifier)",
        color="tab:green",
    )
    bars.set_xticks(xs)
    bars.set_xticklabels([f"sample {i + 1}" for i in xs])
    bars.set_ylabel("Unsupported-claim count")
    bars.set_title(
        "Verifier catches every planted fabrication\n"
        "(planted control, not a measured VLM rate)"
    )
    bars.legend()
    bars.grid(True, alpha=0.3, axis="y")
    text.axis("off")
    sample = logs[0]
    free_text = "FREE-FORM (planted):\n\n" + sample["free_form_text"]
    grounded_text = "GROUNDED (BioCoach):\n\n" + "\n".join(
        f"- {bullet}" for bullet in sample["grounded_bullets"]
    )
    text.text(0.0, 1.0, free_text, va="top", family="monospace", color="tab:red", fontsize=9)
    text.text(0.0, 0.55, grounded_text, va="top", family="monospace", color="tab:green", fontsize=9)
    text.set_title("Sample 1: what each path produced", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_grounded_vs_free_form.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    if not args.bundle.is_dir():
        raise SystemExit(f"bundle dir not found: {args.bundle}")
    OUT.mkdir(exist_ok=True)
    fig_per_flag_scatter()
    fig_multi_view(args.bundle)
    fig_vlm_distribution(args.bundle)
    fig_grounded_vs_free_form(args.bundle)
    written = sorted(path.name for path in OUT.glob("*.png"))
    print(f"wrote {len(written)} figures to {OUT}/: {written}")


if __name__ == "__main__":
    main()
