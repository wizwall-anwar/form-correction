"""Compute offline VLM-versus-clinician agreement from skeleton sequences.

Inputs are ``Sample`` objects with [T, J, 3] joints and clinician bins;
outputs are coordinate keyframes, deterministic placeholder scores, kappa
statistics, and bootstrap intervals. This module is offline: rotated views are
SYNTHETIC, and no real VLM or image renderer is called. The report is NEVER
"validity"; it measures only model-vs-clinician reliability and never claims
that a VLM equals human PT ground truth.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

from .config import RANDOM_SEED, VLM_BACKEND, VLM_TIMEOUT_S
from .m00_data import Sample
from .m01_features import project_2d, rotate_view


def render_keyframes(joints: np.ndarray) -> list[np.ndarray]:
    """Return start, lowest-average-knee/hip-height, and end 2D keyframes."""

    array = np.asarray(joints)
    if array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] == 0:
        raise ValueError("joints must have shape [T, J, 3] with T > 0")
    if array.shape[1] < 4:
        raise ValueError("joints must contain hip and knee joints")
    heights = np.mean(array[:, :4, 1], axis=1)
    indices = (0, int(np.argmin(heights)), array.shape[0] - 1)
    return [project_2d(array[index : index + 1])[0].copy() for index in indices]


def vlm_score_stub(keyframes: list[np.ndarray], seed: int) -> int:
    """Return a deterministic offline bin standing in for a real VLM call.

    This stub stands in for a real VLM call behind a 30-second timeout; a real
    backend is not wired up. It hashes coordinate data and the seed to return
    bin 0, 1, or 2 without making a network request.
    """

    if VLM_BACKEND != "placeholder":
        raise RuntimeError("real VLM backend is not wired up")
    if len(keyframes) != 3:
        raise ValueError("keyframes must contain exactly three arrays")
    arrays = [np.asarray(frame) for frame in keyframes]
    if any(frame.ndim != 2 or frame.shape[1] != 2 for frame in arrays):
        raise ValueError("each keyframe must have shape [J, 2]")
    digest = hashlib.sha256(np.concatenate(arrays).astype(np.float64).tobytes())
    digest.update(str(int(seed)).encode("ascii"))
    return int.from_bytes(digest.digest()[:4], "big") % 3


def vlm_score_with_timeout(
    keyframes: list[np.ndarray], seed: int, timeout_s: float = VLM_TIMEOUT_S
) -> int | str:
    """Run the offline stub under the conceptual timeout, returning no-result on failure."""

    try:
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        return vlm_score_stub(keyframes, seed)
    except Exception:
        return "STUB - no result"


def cohen_kappa(
    ratings_a: list[int], ratings_b: list[int], weights: str | None = None
) -> float:
    """Return unweighted or quadratic-weighted Cohen's kappa."""

    if weights not in (None, "quadratic"):
        raise ValueError("weights must be None or 'quadratic'")
    first = np.asarray(ratings_a)
    second = np.asarray(ratings_b)
    if first.ndim != 1 or second.ndim != 1 or first.size != second.size:
        raise ValueError("ratings must be one-dimensional arrays of equal length")
    if first.size == 0:
        raise ValueError("ratings must be non-empty")

    labels = np.unique(np.concatenate((first, second)))
    first_indices = np.searchsorted(labels, first)
    second_indices = np.searchsorted(labels, second)
    confusion = np.zeros((labels.size, labels.size), dtype=float)
    np.add.at(confusion, (first_indices, second_indices), 1.0)
    observed = confusion / first.size
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / first.size**2

    if weights is None:
        observed_agreement = np.trace(observed)
        expected_agreement = np.trace(expected)
    else:
        if labels.size == 1:
            return 1.0
        indices = np.arange(labels.size)
        disagreement = (indices[:, None] - indices[None, :]) ** 2 / (labels.size - 1) ** 2
        observed_agreement = np.sum((1.0 - disagreement) * observed)
        expected_agreement = np.sum((1.0 - disagreement) * expected)

    return 1.0 if expected_agreement == 1.0 else float(
        (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    )


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Return hand-computed Fleiss' kappa for [items, categories] count data."""

    counts = np.asarray(ratings, dtype=float)
    if counts.ndim != 2 or counts.shape[0] == 0:
        raise ValueError("ratings must have shape [n_items, n_categories]")
    raters = counts.sum(axis=1)
    if np.any(raters <= 0) or not np.allclose(raters, raters[0]):
        raise ValueError("each item must have the same positive rater count")
    n_raters = raters[0]
    item_agreement = np.sum(counts * (counts - 1), axis=1) / (n_raters * (n_raters - 1))
    proportions = counts.sum(axis=0) / counts.sum()
    p_bar = float(np.mean(item_agreement))
    p_bar_e = float(np.sum(proportions**2))
    return 1.0 if p_bar_e == 1.0 else (p_bar - p_bar_e) / (1.0 - p_bar_e)


def bootstrap_ci(
    per_subject_values: list[float], seed: int, n_resamples: int = 200
) -> tuple[float, float]:
    """Return a deterministic 2.5th/97.5th percentile bootstrap interval."""

    values = np.asarray(per_subject_values, dtype=float)
    if values.size == 0 or n_resamples <= 0:
        raise ValueError("values must be non-empty and n_resamples must be positive")
    rng = np.random.default_rng(seed)
    resamples = rng.choice(values, size=(n_resamples, values.size), replace=True)
    means = np.mean(resamples, axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def agreement_report(
    samples: list[Sample], view_degs: tuple = (0, 45), seed: int = RANDOM_SEED
) -> dict:
    """Report model-vs-clinician reliability for synthetic rotated views only."""

    report = {"model_vs_clinician_reliability": {}}
    for angle in view_degs:
        model_bins: list[int] = []
        clinician_bins: list[int] = []
        for index, sample in enumerate(samples):
            keyframes = render_keyframes(rotate_view(sample.joints, angle))
            score = vlm_score_with_timeout(keyframes, seed + index)
            if score == "STUB - no result":
                continue
            model_bins.append(score)
            clinician_bins.append(int(sample.clinician_bin))
        metrics = {"sample_count": len(model_bins), "vlm_bins": model_bins, "clinician_bins": clinician_bins}
        metrics["cohen_kappa"] = cohen_kappa(model_bins, clinician_bins) if model_bins else 0.0
        metrics["quadratic_weighted_cohen_kappa"] = (
            cohen_kappa(model_bins, clinician_bins, weights="quadratic") if model_bins else 0.0
        )
        report["model_vs_clinician_reliability"][str(angle)] = metrics
    return report


def _smoke_test() -> bool:
    """Run small deterministic checks without network access."""

    assert abs(cohen_kappa([0, 1, 2, 1], [0, 2, 2, 1]) - 7 / 11) < 1e-12
    assert abs(fleiss_kappa(np.array([[0, 3, 0], [0, 2, 1], [1, 2, 0], [0, 1, 2]])) + 1 / 35) < 1e-12
    joints = np.zeros((3, 10, 3), dtype=float)
    joints[1, :4, 1] = -1.0
    assert len(render_keyframes(joints)) == 3
    from .m00_data import make_synthetic_squats

    report = agreement_report(make_synthetic_squats(3, RANDOM_SEED), view_degs=(0, 45), seed=RANDOM_SEED)
    return set(report) == {"model_vs_clinician_reliability"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print("PASS" if _smoke_test() else "FAIL")
