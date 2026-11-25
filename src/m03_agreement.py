"""Compute VLM-versus-clinician agreement from skeleton sequences.

Inputs are ``Sample`` objects with [T, J, 3] joints and clinician bins;
outputs are stick-figure images, VLM score bins, kappa statistics, and
bootstrap intervals. Views are SYNTHETIC. When a Gemini API key is set
(env var ``GEMINI_API_KEY``) results use a real VLM; otherwise a
deterministic hash stub runs. The report measures model-vs-clinician
reliability, never "validity", and never claims a VLM equals human PT
ground truth.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .config import (
    GEMINI_MODEL_FALLBACKS,
    RANDOM_SEED,
    VLM_BACKEND,
    VLM_CACHE_DIR,
    VLM_ERROR_BODY_PREVIEW_CHARS,
    VLM_RATE_LIMIT_DELAY_S,
    VLM_RETRY_BASE_DELAY_S,
    VLM_RETRY_MAX_ATTEMPTS,
    VLM_RETRY_MAX_DELAY_S,
    VLM_RETRY_STATUSES,
    VLM_TIMEOUT_S,
)
from .m00_data import Sample
from .m01_features import project_2d, rotate_view

_STICK_CONNECTIONS = [
    (6, 7),  # spine_base -> spine_mid
    (7, 8),  # spine_mid -> shoulder_left
    (7, 9),  # spine_mid -> shoulder_right
    (6, 0),  # spine_base -> hip_left
    (6, 1),  # spine_base -> hip_right
    (0, 2),  # hip_left -> knee_left
    (2, 4),  # knee_left -> ankle_left
    (1, 3),  # hip_right -> knee_right
    (3, 5),  # knee_right -> ankle_right
]

_GEMINI_PROMPT = (
    "Three stick-figure keyframes of a squat: start, bottom, end.\n"
    "Rate movement quality as a single digit:\n"
    "0 = poor form (shallow depth, excessive lean, or asymmetry)\n"
    "1 = moderate form\n"
    "2 = good form\n"
    "Respond with ONLY the digit 0, 1, or 2."
)

_LOGGER = logging.getLogger(__name__)
_STUB_RESULT = "STUB - no result"


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


def render_keyframes_image(keyframes: list[np.ndarray]) -> bytes:
    """Render 2D keyframes as a stick-figure triptych PNG."""

    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    titles = ["Start", "Bottom", "End"]
    fig, axes = plt.subplots(1, len(keyframes), figsize=(3 * len(keyframes), 4))
    if len(keyframes) == 1:
        axes = [axes]
    for idx, (ax, kf) in enumerate(zip(axes, keyframes)):
        for i, j in _STICK_CONNECTIONS:
            if i < len(kf) and j < len(kf):
                ax.plot([kf[i, 0], kf[j, 0]], [kf[i, 1], kf[j, 1]], "b-", linewidth=2)
        ax.scatter(kf[:, 0], kf[:, 1], c="red", s=30, zorder=5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(titles[min(idx, 2)])
        ax.axis("off")
    fig.suptitle("Squat keyframes (SYNTHETIC view)", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def vlm_score_stub(keyframes: list[np.ndarray], seed: int) -> int:
    """Return a deterministic offline bin by hashing coordinates and seed."""

    if len(keyframes) != 3:
        raise ValueError("keyframes must contain exactly three arrays")
    arrays = [np.asarray(frame) for frame in keyframes]
    if any(frame.ndim != 2 or frame.shape[1] != 2 for frame in arrays):
        raise ValueError("each keyframe must have shape [J, 2]")
    digest = hashlib.sha256(np.concatenate(arrays).astype(np.float64).tobytes())
    digest.update(str(int(seed)).encode("ascii"))
    return int.from_bytes(digest.digest()[:4], "big") % 3


def vlm_score_gemini(
    image_bytes: bytes,
    seq_id: str,
    angle: float,
    seed: int,
    rng: random.Random | None = None,
) -> int | str:
    """Call Gemini fallbacks with transient retries, caching successful results."""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        _LOGGER.warning("Gemini scoring failed; GEMINI_API_KEY not set")
        return _STUB_RESULT

    cache_dir = Path(VLM_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{seq_id}_{angle}_{seed}".encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())["bin"]
        except (json.JSONDecodeError, KeyError):
            pass

    payload = json.dumps({
        "contents": [{"parts": [
            {"text": _GEMINI_PROMPT},
            {"inline_data": {
                "mime_type": "image/png",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            }},
        ]}],
    }).encode()

    random_source = rng if rng is not None else random
    failures: list[tuple[str, int | str, str, str]] = []
    for model in GEMINI_MODEL_FALLBACKS:
        status: int | str = "unavailable"
        response_body = ""
        error = "<unavailable>"
        for attempt in range(1, VLM_RETRY_MAX_ATTEMPTS + 1):
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            response_body = ""
            time.sleep(VLM_RATE_LIMIT_DELAY_S)
            try:
                with urllib.request.urlopen(req, timeout=VLM_TIMEOUT_S) as resp:
                    status = getattr(resp, "status", "unavailable")
                    response_body = resp.read().decode("utf-8", errors="replace")
                result = json.loads(response_body)
                parts = result["candidates"][0]["content"]["parts"]
                text = next(
                    part["text"]
                    for part in parts
                    if isinstance(part.get("text"), str)
                )
                stripped_text = text.lstrip()
                if not stripped_text or stripped_text[0] not in ("0", "1", "2"):
                    raise ValueError(f"Gemini response text={text[:40]!r}")
                bin_val = int(stripped_text[0])
                cache_file.write_text(json.dumps({
                    "bin": bin_val,
                    "raw": text,
                    "seq_id": str(seq_id),
                    "angle": float(angle),
                    "model": model,
                }))
                return bin_val
            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    response_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    response_body = ""
                error = f"HTTPError: {exc.reason}"
                if status not in VLM_RETRY_STATUSES or attempt == VLM_RETRY_MAX_ATTEMPTS:
                    break
                retry_after = None
                try:
                    retry_after = float(exc.headers.get("Retry-After"))
                except (AttributeError, TypeError, ValueError):
                    pass
                if retry_after is None or retry_after < 0:
                    delay = min(
                        VLM_RETRY_MAX_DELAY_S,
                        VLM_RETRY_BASE_DELAY_S * 2 ** (attempt - 1),
                    ) * random_source.uniform(0.5, 1.5)
                else:
                    delay = min(VLM_RETRY_MAX_DELAY_S, retry_after)
                time.sleep(delay)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
        failures.append((model, status, response_body, error))

    for model, status, response_body, error in failures:
        safe_model = model.replace(api_key, "[REDACTED]")
        status_text = str(status).replace(api_key, "[REDACTED]")
        preview = response_body.replace(api_key, "[REDACTED]")[
            :VLM_ERROR_BODY_PREVIEW_CHARS
        ]
        safe_error = error.replace(api_key, "[REDACTED]")
        _LOGGER.warning(
            "Gemini model %s failed; HTTP status=%s; response body=%s; error=%s",
            safe_model,
            status_text,
            preview or "<unavailable>",
            safe_error,
        )
    _LOGGER.info(
        "Gemini scoring exhausted all %d fallbacks for seq_id=%s, angle=%s",
        len(GEMINI_MODEL_FALLBACKS),
        str(seq_id).replace(api_key, "[REDACTED]"),
        angle,
    )
    return _STUB_RESULT


def vlm_score_with_timeout(
    keyframes: list[np.ndarray], seed: int, timeout_s: float = VLM_TIMEOUT_S
) -> int | str:
    """Run the offline stub under the conceptual timeout."""

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


def detector_vs_clinician_kappa(
    detector_bins: list[int], clinician_bins: list[int]
) -> dict[str, float | int | None]:
    """Return unweighted and quadratic-weighted Cohen's kappa for the detector.

    A comparison row for Table 4: the deterministic flag-based classifier
    rated as a VLM would be, so model-vs-clinician reliability has a baseline.
    Expects the two lists to align sample-for-sample. Returns ``None`` for the
    kappas when there are fewer than two samples, so an empty row can be
    emitted without crashing. Label is strictly reliability, never validity.
    """

    if len(detector_bins) != len(clinician_bins):
        raise ValueError("detector_bins and clinician_bins must align")
    n = len(detector_bins)
    if n < 2:
        return {"n_scored": n, "cohen_kappa": None,
                "quadratic_weighted_cohen_kappa": None}
    return {
        "n_scored": n,
        "cohen_kappa": cohen_kappa(detector_bins, clinician_bins),
        "quadratic_weighted_cohen_kappa": cohen_kappa(
            detector_bins, clinician_bins, weights="quadratic"
        ),
    }


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
    samples: list[Sample],
    view_degs: tuple = (0, 45),
    seed: int = RANDOM_SEED,
    backend: str | None = None,
) -> dict:
    """Report model-vs-clinician reliability for synthetic rotated views."""

    if backend is None:
        backend = os.environ.get("VLM_BACKEND", VLM_BACKEND)

    report: dict = {"model_vs_clinician_reliability": {}}
    rng = random.Random(seed)
    for angle in view_degs:
        model_bins: list[int] = []
        clinician_bins: list[int] = []
        n_stub = 0
        for index, sample in enumerate(samples):
            rotated = rotate_view(sample.joints, angle)
            keyframes = render_keyframes(rotated)

            if backend == "gemini":
                image = render_keyframes_image(keyframes)
                score = vlm_score_gemini(
                    image, sample.seq_id, angle, seed + index, rng=rng
                )
            else:
                score = vlm_score_with_timeout(keyframes, seed + index)

            if isinstance(score, str):
                n_stub += 1
                continue
            model_bins.append(score)
            clinician_bins.append(int(sample.clinician_bin))

        metrics: dict = {
            "sample_count": len(samples) if backend == "gemini" else len(model_bins),
            "n_scored": len(model_bins),
            "n_stub": n_stub,
            "vlm_bins": model_bins,
            "clinician_bins": clinician_bins,
            "backend": backend,
        }
        if backend == "gemini" and len(model_bins) < 2:
            metrics["cohen_kappa"] = None
            metrics["quadratic_weighted_cohen_kappa"] = None
        elif model_bins:
            metrics["cohen_kappa"] = cohen_kappa(model_bins, clinician_bins)
            metrics["quadratic_weighted_cohen_kappa"] = cohen_kappa(
                model_bins, clinician_bins, weights="quadratic"
            )
        else:
            metrics["cohen_kappa"] = 0.0
            metrics["quadratic_weighted_cohen_kappa"] = 0.0
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

    report = agreement_report(
        make_synthetic_squats(3, RANDOM_SEED), view_degs=(0, 45), seed=RANDOM_SEED
    )
    return set(report) == {"model_vs_clinician_reliability"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print("PASS" if _smoke_test() else "FAIL")
