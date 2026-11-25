"""Repo-wide subject leakage guard for every detector evaluation split."""

import pytest

from src.m01_features import kinematic_flags
from src.m02_detector import _smoke_samples, make_splits, run_evaluation


def test_make_splits_has_no_subject_overlap():
    train, test = make_splits(_smoke_samples(), 11)
    train_subj = [sample.subject_id for sample in train]
    test_subj = [sample.subject_id for sample in test]
    assert len(set(train_subj) & set(test_subj)) == 0


def test_run_evaluation_asserts_disjoint_subject_sets(monkeypatch):
    samples = _smoke_samples()
    flags = [kinematic_flags(sample.joints) for sample in samples]

    monkeypatch.setattr("src.m02_detector.make_splits", lambda rows, seed: (rows, rows))

    with pytest.raises(AssertionError, match="subject leakage"):
        run_evaluation(samples, {0: flags}, 11)
