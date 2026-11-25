"""Test offline agreement metrics, Gemini requests, and failure reporting."""

import io
import json
import logging
import urllib.error

import numpy as np
import pytest

from src import config
from src import m03_agreement
from src.m00_data import make_synthetic_squats
from src.m03_agreement import (
    agreement_report,
    cohen_kappa,
    detector_vs_clinician_kappa,
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


def _http_error(request, status=503, body=b"busy", headers=None):
    """Build a fresh HTTPError with the response fields used by the client."""

    return urllib.error.HTTPError(
        request.full_url,
        status,
        "error",
        headers or {},
        io.BytesIO(body),
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
    cached = json.loads(next(tmp_path.iterdir()).read_text())
    assert cached["model"] == config.GEMINI_MODEL_FALLBACKS[0]

    monkeypatch.setattr(
        m03_agreement.urllib.request,
        "urlopen",
        lambda request, timeout: pytest.fail("cached success made an HTTP request"),
    )
    assert vlm_score_gemini(b"png", "seq", 0, 7) == 2


def test_gemini_retries_503_then_returns_success(monkeypatch, tmp_path):
    requests = []
    sleeps = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if len(requests) < 3:
            raise _http_error(request)
        return _Response(b'{"candidates":[{"content":{"parts":[{"text":"1"}]}}]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement, "GEMINI_MODEL_FALLBACKS", ("model-a",))
    monkeypatch.setattr(m03_agreement.time, "sleep", sleeps.append)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    result = vlm_score_gemini(
        b"png", "seq", 0, 7, rng=m03_agreement.random.Random(7)
    )

    retry_sleeps = [delay for delay in sleeps if delay != config.VLM_RATE_LIMIT_DELAY_S]
    assert result == 1
    assert len(requests) == 3
    assert sleeps.count(config.VLM_RATE_LIMIT_DELAY_S) == 3
    assert len(retry_sleeps) == 2
    assert sum(retry_sleeps) > 0


def test_gemini_falls_back_after_model_exhausts(monkeypatch, tmp_path):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if "model-b" not in request.full_url:
            raise _http_error(request)
        return _Response(b'{"candidates":[{"content":{"parts":[{"text":"2"}]}}]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        m03_agreement, "GEMINI_MODEL_FALLBACKS", ("model-a", "model-b")
    )
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    assert vlm_score_gemini(b"png", "seq", 0, 7) == 2
    assert len(requests) == config.VLM_RETRY_MAX_ATTEMPTS + 1
    assert "model-b" in requests[-1].full_url


def test_gemini_honors_retry_after(monkeypatch, tmp_path):
    calls = 0
    sleeps = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(request, headers={"Retry-After": "2"})
        return _Response(b'{"candidates":[{"content":{"parts":[{"text":"1"}]}}]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement, "GEMINI_MODEL_FALLBACKS", ("model-a",))
    monkeypatch.setattr(m03_agreement.time, "sleep", sleeps.append)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    assert vlm_score_gemini(b"png", "seq", 0, 7) == 1
    assert 2.0 in sleeps


def test_gemini_400_skips_retries_and_tries_next_model(monkeypatch, tmp_path):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise _http_error(request, status=400, body=b"bad request")
        return _Response(b'{"candidates":[{"content":{"parts":[{"text":"2"}]}}]}')

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        m03_agreement, "GEMINI_MODEL_FALLBACKS", ("model-a", "model-b")
    )
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    assert vlm_score_gemini(b"png", "seq", 0, 7) == 2
    assert len(requests) == 2
    assert "model-b" in requests[1].full_url


def test_gemini_exhausts_all_models(monkeypatch, tmp_path, caplog):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        raise _http_error(request)

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level(logging.INFO):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert len(requests) == (
        config.VLM_RETRY_MAX_ATTEMPTS * len(config.GEMINI_MODEL_FALLBACKS)
    )
    assert all(model in caplog.text for model in config.GEMINI_MODEL_FALLBACKS)
    assert "Gemini scoring exhausted all 2 fallbacks for seq_id=seq, angle=0" in caplog.text


@pytest.mark.parametrize(
    ("seq_id", "text", "parts_prefix"),
    [
        ("punctuation", "1.", []),
        ("explanation", "1\nbecause", [{"thought": True}]),
    ],
)
def test_gemini_parses_first_digit_and_searches_parts(
    monkeypatch, tmp_path, seq_id, text, parts_prefix
):
    body = json.dumps({
        "candidates": [{"content": {"parts": parts_prefix + [{"text": text}]}}]
    }).encode()

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        m03_agreement.urllib.request, "urlopen", lambda request, timeout: _Response(body)
    )

    assert vlm_score_gemini(b"png", seq_id, 0, 7) == 1


def test_gemini_200_without_candidates_logs_body_without_key(
    monkeypatch, tmp_path, caplog
):
    api_key = "never-log-this-key"
    body = b'{"blocked":"no candidates"}'

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        m03_agreement.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(body),
    )

    with caplog.at_level(logging.WARNING):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert body.decode() in caplog.text
    assert api_key not in caplog.text


def test_gemini_url_error_logs_exception_type(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        m03_agreement.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(
            urllib.error.URLError("offline")
        ),
    )

    with caplog.at_level(logging.WARNING):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert "URLError" in caplog.text


def test_gemini_invalid_text_logs_text(monkeypatch, tmp_path, caplog):
    body = b'{"candidates":[{"content":{"parts":[{"text":"Maybe"}]}}]}'

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        m03_agreement.urllib.request,
        "urlopen",
        lambda request, timeout: _Response(body),
    )

    with caplog.at_level(logging.WARNING):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert "Maybe" in caplog.text


def test_gemini_missing_key_logs_distinct_message(monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING):
        result = vlm_score_gemini(b"png", "seq", 0, 7)

    assert result == "STUB - no result"
    assert "GEMINI_API_KEY not set" in caplog.text
    assert "HTTP status" not in caplog.text


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
        m03_agreement,
        "vlm_score_gemini",
        lambda *args, **kwargs: "STUB - no result",
    )

    report = agreement_report(samples, view_degs=(0,), seed=7, backend="gemini")
    metrics = report["model_vs_clinician_reliability"]["0"]

    assert metrics["n_stub"] == metrics["sample_count"] == len(samples)
    assert metrics["n_scored"] == 0
    assert metrics["vlm_bins"] == []
    assert metrics["clinician_bins"] == []
    assert metrics["cohen_kappa"] is None
    assert metrics["quadratic_weighted_cohen_kappa"] is None


def test_gemini_agreement_failures_are_seeded_and_unavailable(
    monkeypatch, tmp_path
):
    samples = make_synthetic_squats(2, 7)
    sleep_runs = []

    def fake_urlopen(request, timeout):
        raise _http_error(request)

    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setattr(m03_agreement, "VLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(m03_agreement, "render_keyframes_image", lambda _: b"png")
    monkeypatch.setattr(m03_agreement.urllib.request, "urlopen", fake_urlopen)

    reports = []
    for _ in range(2):
        sleeps = []
        monkeypatch.setattr(m03_agreement.time, "sleep", sleeps.append)
        reports.append(
            agreement_report(samples, view_degs=(0,), seed=7, backend="gemini")
        )
        sleep_runs.append(sleeps)

    metrics = reports[0]["model_vs_clinician_reliability"]["0"]
    assert reports[0] == reports[1]
    assert sleep_runs[0] == sleep_runs[1]
    assert metrics["n_scored"] == 0
    assert metrics["n_stub"] == 2
    assert metrics["cohen_kappa"] is None
    assert metrics["quadratic_weighted_cohen_kappa"] is None


def test_gemini_model_is_not_retired_flash_lite():
    assert config.GEMINI_MODEL_ID != "gemini-2.0-flash-lite"
    assert config.GEMINI_MODEL_FALLBACKS[0] == config.GEMINI_MODEL_ID


def test_detector_vs_clinician_kappa_matches_cohen_on_equal_length_rows():
    """Helper returns Cohen's kappas and the sample count for aligned rows."""
    result = detector_vs_clinician_kappa([0, 1, 2, 1], [0, 2, 2, 1])
    assert result["n_scored"] == 4
    assert abs(result["cohen_kappa"] - cohen_kappa([0, 1, 2, 1], [0, 2, 2, 1])) < 1e-12
    assert abs(result["quadratic_weighted_cohen_kappa"]
               - cohen_kappa([0, 1, 2, 1], [0, 2, 2, 1], weights="quadratic")) < 1e-12


def test_detector_vs_clinician_kappa_handles_small_samples():
    """Fewer than two samples should return None kappas without crashing."""
    result = detector_vs_clinician_kappa([0], [1])
    assert result == {"n_scored": 1, "cohen_kappa": None,
                      "quadratic_weighted_cohen_kappa": None}


def test_detector_vs_clinician_kappa_rejects_misaligned_rows():
    """Mismatched lengths raise ValueError rather than silently truncating."""
    with pytest.raises(ValueError):
        detector_vs_clinician_kappa([0, 1, 2], [0, 1])
