"""Test the grounded critique demo inputs, outputs, and claim checks.

Inputs are small detector records and text samples. Outputs are verified
sentences and a JSONL comparison record. These tests exist to keep the demo
deterministic and measurement-grounded.
"""

import json

from app.biocoach_demo import (
    BANNED_TERMS,
    format_bullets,
    log_hallucination_comparison,
    verify_claims,
)


def _detector(abstain=False):
    return {
        "flags": {
            "knee_flexion_min_deg": 82.4,
            "knee_depth_pass": False,
            "trunk_lean_max_deg": 18.2,
            "trunk_lean_pass": True,
            "symmetry_diff_deg": 7.1,
            "symmetry_pass": False,
        },
        "abstain_flag": abstain,
        "confidence": 0.91,
        "view_deg": 0.0,
    }


def test_abstain_returns_one_refusal_line():
    bullets = format_bullets(_detector(abstain=True))
    assert len(bullets) == 1
    assert "unreliable" in bullets[0].lower()


def test_non_abstain_returns_three_clean_bullets():
    bullets = format_bullets(_detector())
    assert len(bullets) == 3
    lowered = " ".join(bullets).lower()
    assert not any(term in lowered for term in BANNED_TERMS)


def test_verifier_replaces_fabricated_number():
    detector = _detector()
    bullets = format_bullets(detector)
    verified = verify_claims(["Knee depth measured 999.0 degrees.", *bullets[1:]], detector)
    assert verified[0] == bullets[0]


def test_verifier_replaces_planted_banned_word():
    detector = _detector()
    bullets = format_bullets(detector)
    planted = f"Knee depth measured 82.4 degrees; {BANNED_TERMS[0]} note."
    verified = verify_claims([planted, *bullets[1:]], detector)
    assert verified[0] == bullets[0]


def test_log_counts_unsupported_free_form_number(tmp_path):
    detector = _detector()
    grounded = format_bullets(detector)
    path = tmp_path / "results" / "halluc_log.jsonl"
    logged = log_hallucination_comparison(
        "Knee is 82.4 degrees and another measure is 123.0 degrees.",
        grounded,
        detector,
        str(path),
    )
    assert logged["free_form_unsupported_count"] == 1
    assert logged["grounded_unsupported_count"] == 0
    assert json.loads(path.read_text(encoding="utf-8")) == logged
