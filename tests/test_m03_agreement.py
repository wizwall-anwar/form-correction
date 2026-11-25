"""Test offline agreement metrics, Gemini requests, and failure reporting."""

import io
import logging
import urllib.error

import numpy as np

from src import config
from src import m03_agreement
from src.m00_data import make_synthetic_squats
from src.m03_agreement import (
    agreement_report,
    cohen_kappa,
    fleiss_kappa,
    render_keyframes,
    render_keyframes_image,
    vlm_score_gemini,
    vlm_score_with_timeout,
)


class _Response:
    """Provide the small urlopen response interface used by the Gemini client."""

    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


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


def test_render_keyframes_image_returns_png_bytes():
    joints = np.zeros((3, 10, 3), dtype=float)
    joints[:, 0] = (-0.2, 1.0, 0.0)
    joints[:, 1] = (0.2, 1.0, 0.0)
    joints[1, :4, 1] = -1.0
    keyframes = render_keyframes(joints)
    image = render_keyframes_image(keyframes)
    assert isinstance(image, bytes)
    assert image[:4] == b"\x89PNG"


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


def test_gemini_parses_digit_and_sends_key_only_in_header(monkeypatch, tmp_path):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return _Response(b'{"candidates":[{"content":{"parts":[{"text":"2"}]}}]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    assert vlm_score_gemini(b"png", "seq", 0, 7) == 2
    assert requests[0].get_header("X-goog-api-key") == "test-secret"
    assert "test-secret" not in requests[0].full_url


def test_gemini_http_404_returns_stub_and_logs_without_key(
    monkeypatch, tmp_path, caplog
):
    api_key = "never-log-this-key"

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b'model retired'),
        )

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.WARNING):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert "404" in caplog.text
    assert "model retired" in caplog.text
    assert api_key not in caplog.text
    assert not list(tmp_path.iterdir())


def test_gemini_failures_are_excluded_from_agreement(monkeypatch, tmp_path):
    samples = make_synthetic_squats(3, 7)
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        m03_agreement, "vlm_score_gemini", lambda *args: "STUB - no result"
    )

    report = agreement_report(samples, view_degs=(0,), seed=7, backend="gemini")
    metrics = report["model_vs_clinician_reliability"]["0"]

    assert metrics["n_stub"] == metrics["sample_count"] == len(samples)
    assert metrics["n_scored"] == 0
    assert metrics["vlm_bins"] == []
    assert metrics["clinician_bins"] == []
    assert metrics["cohen_kappa"] is None
    assert metrics["quadratic_weighted_cohen_kappa"] is None


def test_gemini_model_is_not_retired_flash_lite():
    assert config.GEMINI_MODEL_ID != "gemini-2.0-flash-lite"
