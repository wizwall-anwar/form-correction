"""Tests for subject-safe detector splits and the tiny offline pipeline."""

from src.m02_detector import (
    _smoke_samples,
    calibrate,
    evaluate,
    make_clip_split,
    make_splits,
    predict_with_abstention,
    risk_coverage,
    train_baseline,
)


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
