"""Repo-wide subject leakage guard for the detector split."""

from src.m02_detector import _smoke_samples, make_splits


def test_make_splits_has_no_subject_overlap():
    train, test = make_splits(_smoke_samples(), 11)
    train_subj = [sample.subject_id for sample in train]
    test_subj = [sample.subject_id for sample in test]
    assert len(set(train_subj) & set(test_subj)) == 0
