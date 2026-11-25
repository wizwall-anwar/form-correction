"""Train and evaluate a small subject-safe detector from kinematic flags.

Inputs are ``Sample`` objects with standardized joints and clinician scores.
Outputs are held-out regression metrics, classifier calibration metrics, and
risk-coverage data. These small NumPy baselines keep the offline evaluation
reproducible; temporal deep learning is out of scope.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import numpy as np

from .config import (
    ABSTAIN_CONFIDENCE_THRESHOLD,
    BOOTSTRAP_RESAMPLES,
    CLINICIAN_SCORE_BIN_QUANTILES,
    RIDGE_ALPHA,
    SELECTIVE_COVERAGES,
)
from .m00_data import Sample
from .m01_features import flag_error, kinematic_flags


class _NumpyLogisticRegression:
    """Small multinomial classifier exposing the expected prediction interface."""

    def __init__(self, weights: np.ndarray, classes: np.ndarray) -> None:
        self.weights = weights
        self.classes_ = classes
        self.temperature = 1.0

    def _logits(self, features: np.ndarray) -> np.ndarray:
        design = np.column_stack((np.ones(len(features)), features))
        return design @ self.weights

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.empty((0, len(self.classes_)))
        logits = self._logits(features) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / np.sum(probabilities, axis=1, keepdims=True)


class _RidgeRegressor:
    """Closed-form ridge model with training-set feature scaling."""

    def __init__(self, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                 target_mean: float) -> None:
        self.weights = weights
        self.mean = mean
        self.scale = scale
        self.target_mean = target_mean

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(features)), standardized))
        return design @ self.weights


def make_splits(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    """Split samples by shuffled subject into deterministic train and test groups."""

    subjects = np.array(sorted({sample.subject_id for sample in samples}), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    if len(subjects) < 2:
        train_subjects, test_subjects = set(subjects), set()
    else:
        n_train = min(len(subjects) - 1, max(1, round(0.7 * len(subjects))))
        train_subjects, test_subjects = set(subjects[:n_train]), set(subjects[n_train:])
    train = [sample for sample in samples if sample.subject_id in train_subjects]
    test = [sample for sample in samples if sample.subject_id in test_subjects]
    assert not train_subjects & test_subjects, "subject leakage between train and test splits"
    return train, test


def make_clip_split(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    """Split clips while deliberately ignoring subjects; this is leaky and never for evaluation."""

    order = np.random.default_rng(seed).permutation(len(samples))
    n_train = 0 if not samples else min(len(samples) - 1, max(1, round(0.7 * len(samples))))
    return [samples[i] for i in order[:n_train]], [samples[i] for i in order[n_train:]]


def extract_features(samples: list[Sample]) -> np.ndarray:
    """Return the three numeric kinematic flag measurements as an ``[N, 3]`` array."""

    keys = ("knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg")
    rows = [[kinematic_flags(sample.joints)[key] for key in keys] for sample in samples]
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _features_from_flags(flags: list[dict]) -> np.ndarray:
    keys = ("knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg")
    return np.asarray([[row[key] for key in keys] for row in flags], dtype=float).reshape(-1, 3)


def train_tertile_edges(train: list[Sample]) -> tuple[float, float]:
    """Compute score-bin cuts from training-subject scores only."""

    scores = np.asarray([sample.clinician_score for sample in train
                         if sample.clinician_score is not None], dtype=float)
    if not len(scores):
        raise ValueError("cannot compute tertiles without clinician scores")
    distinct = np.unique(scores)
    if len(distinct) == 1:
        return float("inf"), float("inf")
    if len(distinct) == 2:
        return float(np.mean(distinct)), float("inf")
    edges = np.quantile(scores, CLINICIAN_SCORE_BIN_QUANTILES)
    return float(edges[0]), float(edges[1])


def rebin(samples: list[Sample], edges: tuple[float, float]) -> list[Sample]:
    """Return copies with training-derived score bins applied."""

    return [
        replace(
            sample,
            clinician_bin=0 if sample.clinician_score is None
            else int(np.digitize(sample.clinician_score, edges)),
        )
        for sample in samples
    ]


def train_regressor(train: list[Sample], seed: int) -> _RidgeRegressor:
    """Fit closed-form ridge regression to raw clinician scores."""

    del seed
    scored = [sample for sample in train if sample.clinician_score is not None]
    if not scored:
        raise ValueError("cannot train on samples without clinician scores")
    features = extract_features(scored)
    targets = np.asarray([sample.clinician_score for sample in scored], dtype=float)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return _RidgeRegressor(weights, mean, scale, float(np.mean(targets)))


def predict_scores(model: _RidgeRegressor, samples: list[Sample]) -> list[float]:
    """Predict raw clinician scores from the three flag measurements."""

    return [float(value) for value in model.predict(extract_features(samples))]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_rho(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Return Spearman rho with average tie ranks, or zero for a constant input."""

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or not len(left):
        raise ValueError("inputs must be non-empty and have matching shapes")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return 0.0
    return round(float(np.corrcoef(left_ranks, right_ranks)[0, 1]), 12)


def mean_absolute_error(a: list[float] | np.ndarray,
                        b: list[float] | np.ndarray) -> float:
    """Return the mean absolute difference between matching values."""

    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or not len(left):
        raise ValueError("inputs must be non-empty and have matching shapes")
    return float(np.mean(np.abs(left - right)))


def evaluate_regression(model: _RidgeRegressor, samples: list[Sample]) -> dict:
    """Evaluate raw-score predictions and the training-mean MAE baseline."""

    scored = [sample for sample in samples if sample.clinician_score is not None]
    true = [float(sample.clinician_score) for sample in scored]
    predicted = predict_scores(model, scored)
    baseline = [model.target_mean] * len(scored)
    return {
        "spearman_rho": spearman_rho(true, predicted),
        "mae": mean_absolute_error(true, predicted),
        "mae_mean_baseline": mean_absolute_error(true, baseline),
        "n": len(scored),
    }


def bootstrap_ci(true: list[float], pred: list[float], seed: int,
                 subject_ids: list[str] | None = None) -> dict:
    """Return deterministic percentile CIs by resampling whole test subjects."""

    if len(true) != len(pred) or not true:
        raise ValueError("true and pred must be non-empty and have matching lengths")
    subjects = subject_ids or [str(index) for index in range(len(true))]
    if len(subjects) != len(true):
        raise ValueError("subject_ids must align with true and pred")
    unique_subjects = list(dict.fromkeys(subjects))
    rows_by_subject = {subject: [i for i, value in enumerate(subjects) if value == subject]
                       for subject in unique_subjects}
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    maes: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = [index for subject in drawn for index in rows_by_subject[str(subject)]]
        sampled_true = [true[index] for index in indices]
        sampled_pred = [pred[index] for index in indices]
        rhos.append(spearman_rho(sampled_true, sampled_pred))
        maes.append(mean_absolute_error(sampled_true, sampled_pred))
    return {
        "spearman_rho_ci95": [float(value) for value in np.percentile(rhos, [2.5, 97.5])],
        "mae_ci95": [float(value) for value in np.percentile(maes, [2.5, 97.5])],
    }


def train_baseline(train_samples: list[Sample], seed: int) -> Any:
    """Fit a deterministic logistic-regression classifier on flag measurements."""

    features = extract_features(train_samples)
    labels = np.asarray([sample.clinician_bin for sample in train_samples])
    if not len(labels):
        raise ValueError("cannot train on empty samples")
    classes = np.unique(labels)
    class_indices = np.searchsorted(classes, labels)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, (features.shape[1] + 1, len(classes)))
    design = np.column_stack((np.ones(len(features)), features))
    targets = np.eye(len(classes))[class_indices]
    for _ in range(80):
        logits = design @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        weights -= 0.01 * (design.T @ (probabilities - targets)) / len(features)
    return _NumpyLogisticRegression(weights, classes)


def calibrate(model: Any, val_samples: list[Sample]) -> Any:
    """Calibrate a fitted model on a held-out subject group, falling back if unusable."""

    if not val_samples:
        return model
    try:
        features = extract_features(val_samples)
        labels = [sample.clinician_bin for sample in val_samples]
        class_indices = {label: index for index, label in enumerate(model.classes_)}
        target_indices = [class_indices[label] for label in labels]
        logits = model._logits(features)
        best_temperature, best_nll = 1.0, float("inf")
        for temperature in np.linspace(0.5, 3.0, 20):
            scaled = logits / temperature
            scaled -= np.max(scaled, axis=1, keepdims=True)
            probabilities = np.exp(scaled)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            nll = -float(np.mean(np.log(probabilities[np.arange(len(labels)), target_indices] + 1e-12)))
            if nll < best_nll:
                best_temperature, best_nll = float(temperature), nll
        model.temperature = best_temperature
        return model
    except (KeyError, ValueError, TypeError):
        return model


def predict_with_abstention(model: Any, samples: list[Sample], confidence_threshold: float = ABSTAIN_CONFIDENCE_THRESHOLD) -> list[dict]:
    """Predict bins and confidence, abstaining below the configured threshold."""

    probabilities = model.predict_proba(extract_features(samples))
    classes = np.asarray(model.classes_)
    return [
        {
            "pred_bin": int(classes[int(np.argmax(probability))]),
            "confidence": float(np.max(probability)),
            "abstain_flag": bool(np.max(probability) < confidence_threshold),
        }
        for probability in probabilities
    ]


def risk_coverage(predictions: list[dict], true_bins: list[int]) -> dict:
    """Sweep actual confidence thresholds and return selective risk by coverage."""

    if not predictions:
        return {"coverage": [], "risk": []}
    ordered = sorted(enumerate(predictions), key=lambda item: item[1]["confidence"], reverse=True)
    coverage, risk, included = [], [], []
    for index, (original_index, prediction) in enumerate(ordered):
        included.append(original_index)
        if index == len(ordered) - 1 or prediction["confidence"] != ordered[index + 1][1]["confidence"]:
            coverage.append(len(included) / len(predictions))
            risk.append(float(np.mean([predictions[i]["pred_bin"] != true_bins[i] for i in included])))
    return {"coverage": coverage, "risk": risk}


def evaluate(predictions: list[dict], true_bins: list[int]) -> dict:
    """Return macro F1, accuracy, and binned ECE among non-abstained predictions."""

    kept = [i for i, prediction in enumerate(predictions) if not prediction["abstain_flag"]]
    if not kept:
        return {"macro_f1": 0.0, "accuracy": 0.0, "ece": 0.0}
    predicted = [predictions[i]["pred_bin"] for i in kept]
    actual = [true_bins[i] for i in kept]
    confidence = np.asarray([predictions[i]["confidence"] for i in kept])
    correct = np.asarray(predicted) == np.asarray(actual)
    f1_scores = []
    for label in np.unique(actual):
        true_positive = sum(p == label and a == label for p, a in zip(predicted, actual))
        predicted_count = sum(p == label for p in predicted)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / sum(a == label for a in actual)
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(edges[:-1], edges[1:]):
        members = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if np.any(members):
            ece += np.sum(members) / len(kept) * abs(np.mean(correct[members]) - np.mean(confidence[members]))
    return {
        "macro_f1": float(np.mean(f1_scores)),
        "accuracy": float(np.mean(correct)),
        "ece": float(ece),
    }


def selective_accuracy_at_coverage(
    predictions: list[dict],
    true_bins: list[int],
    coverages: tuple[float, ...] = SELECTIVE_COVERAGES,
) -> dict[str, float]:
    """Return selective accuracy, not class precision, for top-confidence rows."""

    if len(predictions) != len(true_bins):
        raise ValueError("predictions and true_bins must align")
    if not predictions:
        return {str(coverage): 0.0 for coverage in coverages}
    order = sorted(range(len(predictions)),
                   key=lambda index: predictions[index]["confidence"], reverse=True)
    results: dict[str, float] = {}
    for coverage in coverages:
        keep = min(len(order), max(1, int(np.ceil(coverage * len(order)))))
        indices = order[:keep]
        results[str(coverage)] = float(np.mean([
            predictions[index]["pred_bin"] == true_bins[index] for index in indices
        ]))
    return results


def _subject_ids(samples: list[Sample]) -> set[str]:
    return {sample.subject_id for sample in samples}


def run_evaluation(samples: list[Sample], flags_by_angle: dict, seed: int) -> dict:
    """Run held-out regression and calibrated classification without subject leakage."""

    train, test = make_splits(samples, seed)
    fit, val = make_splits(train, seed)
    train_subjects = _subject_ids(train)
    test_subjects = _subject_ids(test)
    fit_subjects = _subject_ids(fit)
    val_subjects = _subject_ids(val)
    assert len(train_subjects & test_subjects) == 0, "subject leakage between train and test"
    assert len(fit_subjects & val_subjects) == 0, "subject leakage between fit and validation"
    assert len(fit_subjects & test_subjects) == 0, "subject leakage between fit and test"
    assert len(val_subjects & test_subjects) == 0, "subject leakage between validation and test"

    edges = train_tertile_edges(train)
    train_binned = rebin(train, edges)
    fit_binned = rebin(fit, edges)
    val_binned = rebin(val, edges)
    test_binned = rebin(test, edges)

    regressor = train_regressor(train_binned, seed)
    subject_metrics = evaluate_regression(regressor, test_binned)
    subject_true = [float(sample.clinician_score) for sample in test_binned
                    if sample.clinician_score is not None]
    subject_pred = predict_scores(
        regressor,
        [sample for sample in test_binned if sample.clinician_score is not None],
    )
    subject_ids = [sample.subject_id for sample in test_binned
                   if sample.clinician_score is not None]
    subject_metrics.update(bootstrap_ci(subject_true, subject_pred, seed, subject_ids))
    subject_metrics["n_test"] = subject_metrics.pop("n")

    classifier = calibrate(train_baseline(fit_binned, seed), val_binned)
    test_predictions = predict_with_abstention(classifier, test_binned)
    test_bins = [sample.clinician_bin for sample in test_binned]
    classifier_metrics = evaluate(test_predictions, test_bins)
    selective_risk = risk_coverage(test_predictions, test_bins)
    selective_accuracy = selective_accuracy_at_coverage(test_predictions, test_bins)
    class_counts = {str(label): int(test_bins.count(label)) for label in sorted(set(test_bins))}

    clip_train, clip_test = make_clip_split(samples, seed)
    clip_regressor = train_regressor(clip_train, seed)
    clip_metrics = evaluate_regression(clip_regressor, clip_test)
    clip_metrics = {
        "spearman_rho": clip_metrics["spearman_rho"],
        "mae": clip_metrics["mae"],
        "n_test": clip_metrics["n"],
    }

    sample_indices = {id(sample): index for index, sample in enumerate(samples)}
    test_indices = [sample_indices[id(sample)] for sample in test]
    reference_flags = flags_by_angle.get(0)
    if reference_flags is None or len(reference_flags) != len(samples):
        raise ValueError("flags_by_angle[0] must align with samples")
    table2_rows = []
    for angle in sorted(flags_by_angle):
        angle_flags = flags_by_angle[angle]
        if len(angle_flags) != len(samples):
            raise ValueError("each angle's flags must align with samples")
        held_out_flags = [angle_flags[index] for index in test_indices]
        angle_pred = [float(value) for value in regressor.predict(_features_from_flags(held_out_flags))]
        errors = [
            value
            for index in test_indices
            for value in flag_error(angle_flags[index], reference_flags[index]).values()
        ]
        table2_rows.append({
            "angle": float(angle),
            "spearman_rho": spearman_rho(subject_true, angle_pred),
            "mae": mean_absolute_error(subject_true, angle_pred),
            "flag_error": float(np.mean(errors)) if errors else 0.0,
        })

    sequence_counts = {
        subject: sum(sample.subject_id == subject for sample in samples)
        for subject in _subject_ids(samples)
    }
    all_predictions = predict_with_abstention(classifier, rebin(samples, edges))
    return {
        "subject_split": subject_metrics,
        "clip_split": clip_metrics,
        "ece": classifier_metrics["ece"],
        "risk_coverage": selective_risk,
        "selective_accuracy_at_coverage": selective_accuracy,
        "test_class_counts": class_counts,
        "table2_rows": table2_rows,
        "max_sequences_per_subject": max(sequence_counts.values(), default=0),
        "n_subjects": len(sequence_counts),
        "n_sequences": len(samples),
        "n_test": len(test_binned),
        "predictions": all_predictions,
    }


def _smoke_samples() -> list[Sample]:
    """Build eight tiny labeled samples for the offline smoke test."""

    labels = ((0, 1), (1, 2), (2, 0), (0, 1))
    samples = []
    for subject, subject_labels in enumerate(labels):
        for clip, label in enumerate(subject_labels):
            joints = np.zeros((3, 10, 3), dtype=float)
            joints[:, 0] = (-0.2, 1.0, 0.0)
            joints[:, 1] = (0.2, 1.0, 0.0)
            joints[:, 2] = (-0.2, 0.5 + 0.02 * label, 0.0)
            joints[:, 3] = (0.2, 0.5 + 0.02 * label, 0.0)
            joints[:, 4] = (-0.2, 0.0, 0.0)
            joints[:, 5] = (0.2, 0.0, 0.0)
            joints[:, 6] = (0.0, 0.9, 0.0)
            joints[:, 7] = (0.0, 1.8, 0.0)
            score = 0.5 + 0.07 * (subject * 2 + clip)
            samples.append(Sample(f"s{subject}", None, f"q{subject}-{clip}", "squat", joints, 30.0, score, label, {}, 0.0, False, "synthetic"))
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        try:
            samples = _smoke_samples()
            flags = [kinematic_flags(sample.joints) for sample in samples]
            run_evaluation(samples, {0: flags}, 11)
            print("PASS")
        except Exception as error:
            print(f"FAIL: {error}")
