import numpy as np

from src.m00_data import make_synthetic_squats
from src.m03_agreement import (
    agreement_report,
    cohen_kappa,
    fleiss_kappa,
    vlm_score_with_timeout,
)


def test_cohen_kappa_matches_hand_computed_value():
    np.testing.assert_allclose(cohen_kappa([0, 1, 2, 1], [0, 2, 2, 1]), 7 / 11, atol=1e-12)


def test_fleiss_kappa_matches_hand_computed_value():
    ratings = np.array([[0, 3, 0], [0, 2, 1], [1, 2, 0], [0, 1, 2]])
    np.testing.assert_allclose(fleiss_kappa(ratings), -1 / 35, atol=1e-12)


def test_vlm_stub_returns_int_or_explicit_no_result():
    keyframes = [np.zeros((10, 2)) for _ in range(3)]
    result = vlm_score_with_timeout(keyframes, seed=7)
    assert isinstance(result, (int, str))
    assert result == "STUB - no result" or result in (0, 1, 2)


def test_agreement_report_describes_reliability_without_validity_claims():
    report = agreement_report(make_synthetic_squats(3, 7), view_degs=(0, 45), seed=7)

    def keys(value):
        if isinstance(value, dict):
            return list(value) + [key for item in value.values() for key in keys(item)]
        if isinstance(value, list):
            return [key for item in value for key in keys(item)]
        return []

    assert "model_vs_clinician_reliability" in report
    assert all("validity" not in str(key).lower() for key in keys(report))
