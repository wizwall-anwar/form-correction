"""Tests for subject-safe detector splits and the tiny offline pipeline."""

from dataclasses import replace
import json

import numpy as np

from src.m02_detector import (
    _smoke_samples,
    bootstrap_ci,
    calibrate,
    evaluate,
    evaluate_regression,
    make_clip_split,
    make_splits,
    mean_absolute_error,
    per_flag_correlations,
    predict_scores,
    predict_with_abstention,
    risk_coverage,
    run_evaluation,
    selective_accuracy_at_coverage,
    spearman_rho,
    train_regressor,
    train_tertile_edges,
    rebin,
    train_baseline,
)
from src.m01_features import kinematic_flags


def test_splits_and_leaky_clip_split():
    samples = _smoke_samples()
    train, test = make_splits(samples, 11)
    assert not {sample.subject_id for sample in train} & {sample.subject_id for sample in test}
    # The clip split intentionally ignores subject_id and may overlap subjects.
    clip_train, clip_test = make_clip_split(samples, 11)
    assert {sample.subject_id for sample in clip_train} & {sample.subject_id for sample in clip_test}


def test_risk_coverage_increases_as_threshold_drops():
    predictions = [
        {"pred_bin": 0, "confidence": 0.9, "abstain_flag": False},
        {"pred_bin": 1, "confidence": 0.6, "abstain_flag": True},
        {"pred_bin": 2, "confidence": 0.3, "abstain_flag": True},
    ]
    coverage = risk_coverage(predictions, [0, 2, 1])["coverage"]
    assert coverage == sorted(coverage)


def test_smoke_pipeline():
    samples = _smoke_samples()
    train, test = make_splits(samples, 11)
    held_out = sorted({sample.subject_id for sample in train})[-1]
    fit = [sample for sample in train if sample.subject_id != held_out]
    val = [sample for sample in train if sample.subject_id == held_out]
    model = calibrate(train_baseline(fit, 11), val)
    predictions = predict_with_abstention(model, test)
    assert len(predictions) == len(test)
    assert set(evaluate(predictions, [sample.clinician_bin for sample in test])) >= {"macro_f1", "ece"}
    assert len(risk_coverage(predictions, [sample.clinician_bin for sample in test])["coverage"]) > 0


def test_train_tertile_edges_ignore_test_scores() -> None:
    samples = _smoke_samples()[:4]
    train = [replace(sample, clinician_score=score) for sample, score in zip(samples[:3], [0.5, 0.7, 0.9])]
    test = [replace(samples[3], clinician_score=100.0)]

    edges = train_tertile_edges(train)
    rebinned = rebin(train + test, edges)

    assert np.allclose(edges, np.quantile([0.5, 0.7, 0.9], [1 / 3, 2 / 3]))
    assert rebinned[-1].clinician_bin == 2


def test_regression_metrics_have_hand_checked_values() -> None:
    assert spearman_rho([1, 2, 3, 4], [1, 3, 2, 4]) == 0.8
    assert spearman_rho([1, 2, 2, 4], [1, 2, 2, 4]) == 1.0
    assert spearman_rho([1, 1, 1], [1, 2, 3]) == 0.0
    assert mean_absolute_error([1, 3, 8], [2, 5, 5]) == 2.0


def test_ridge_recovers_clean_linear_relation(monkeypatch) -> None:
    samples = [replace(sample, clinician_score=float(2 * index + 1))
               for index, sample in enumerate(_smoke_samples()[:6])]
    features = {sample.seq_id: np.array([index, 2 * index, -index], dtype=float)
                for index, sample in enumerate(samples)}
    monkeypatch.setattr(
        "src.m02_detector.extract_features",
        lambda rows: np.asarray([features[row.seq_id] for row in rows]),
    )

    model = train_regressor(samples, 11)
    predictions = predict_scores(model, samples)

    assert spearman_rho([sample.clinician_score for sample in samples], predictions) == 1.0
    assert evaluate_regression(model, samples)["mae"] < 0.5


def test_bootstrap_ci_is_deterministic_for_seed() -> None:
    true = [0.5, 0.6, 0.8, 0.9]
    pred = [0.55, 0.7, 0.75, 0.95]

    assert bootstrap_ci(true, pred, 19) == bootstrap_ci(true, pred, 19)


def test_selective_accuracy_degrades_as_lower_confidence_rows_are_added() -> None:
    predictions = [
        {"pred_bin": 0, "confidence": 0.9, "abstain_flag": False},
        {"pred_bin": 1, "confidence": 0.8, "abstain_flag": False},
        {"pred_bin": 0, "confidence": 0.2, "abstain_flag": True},
        {"pred_bin": 0, "confidence": 0.1, "abstain_flag": True},
    ]

    accuracy = selective_accuracy_at_coverage(predictions, [0, 1, 2, 2], (0.5, 1.0))

    assert accuracy == {"0.5": 1.0, "1.0": 0.5}


def test_run_evaluation_returns_subject_safe_json_metrics() -> None:
    samples = _smoke_samples()
    flags = [kinematic_flags(sample.joints) for sample in samples]

    result = run_evaluation(samples, {0: flags, 15: flags}, 11)

    assert set(result) >= {
        "subject_split", "per_flag_subject_split", "clip_split", "ece",
        "risk_coverage", "selective_accuracy_at_coverage", "test_class_counts",
        "table2_rows", "max_sequences_per_subject", "n_subjects",
        "n_sequences", "predictions",
    }
    assert set(result["subject_split"]) >= {
        "spearman_rho", "mae", "spearman_rho_ci95", "mae_ci95",
        "mae_mean_baseline", "n_test",
    }
    assert set(result["per_flag_subject_split"]) == {
        "knee_flexion_max_deg", "trunk_lean_max_deg", "symmetry_diff_deg",
    }
    assert len(result["predictions"]) == len(samples)
    assert json.loads(json.dumps(result)) == result


def test_per_flag_correlations_perfectly_rank_a_monotone_feature(monkeypatch) -> None:
    """A flag that monotonically tracks the score yields rho == 1.0."""
    samples = [replace(sample, clinician_score=float(index))
               for index, sample in enumerate(_smoke_samples()[:5])]
    # knee rises with score, trunk is constant (rho == 0), symmetry falls.
    monkeypatch.setattr(
        "src.m02_detector.extract_features",
        lambda rows: np.asarray(
            [[float(sample.clinician_score), 1.0, -float(sample.clinician_score)]
             for sample in rows]
        ),
    )
    correlations = per_flag_correlations(samples)
    assert correlations["knee_flexion_max_deg"] == 1.0
    assert correlations["trunk_lean_max_deg"] == 0.0
    assert correlations["symmetry_diff_deg"] == -1.0


def test_per_flag_correlations_returns_zeros_when_insufficient_scores() -> None:
    """Fewer than two scored samples should not crash; emit zeros for the row."""
    samples = _smoke_samples()[:1]
    samples = [replace(sample, clinician_score=None) for sample in samples]
    correlations = per_flag_correlations(samples)
    assert correlations == {
        "knee_flexion_max_deg": 0.0,
        "trunk_lean_max_deg": 0.0,
        "symmetry_diff_deg": 0.0,
    }
