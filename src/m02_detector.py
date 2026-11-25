"""Train and evaluate a small subject-safe detector from kinematic flags.

Inputs are ``Sample`` objects with standardized joints and clinician bins.
Outputs are classifier predictions, calibration metrics, and risk-coverage data.
This baseline exists as the offline detector; temporal deep learning is out of scope.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .config import ABSTAIN_CONFIDENCE_THRESHOLD
from .m00_data import Sample
from .m01_features import kinematic_flags


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
            samples.append(Sample(f"s{subject}", None, f"q{subject}-{clip}", "squat", joints, 30.0, None, label, {}, 0.0, False, "synthetic"))
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        try:
            samples = _smoke_samples()
            train, test = make_splits(samples, 11)
            held_out = sorted({sample.subject_id for sample in train})[-1]
            fit = [sample for sample in train if sample.subject_id != held_out]
            val = [sample for sample in train if sample.subject_id == held_out]
            model = calibrate(train_baseline(fit, 11), val)
            predictions = predict_with_abstention(model, test)
            risk_coverage(predictions, [sample.clinician_bin for sample in test])
            evaluate(predictions, [sample.clinician_bin for sample in test])
            print("PASS")
        except Exception as error:
            print(f"FAIL: {error}")
