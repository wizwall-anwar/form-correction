"""Produce a deterministic, measurement-only critique from detector output.

Inputs are one detector record containing kinematic flags and reliability
fields. Outputs are three verified sentences, one refusal sentence when the
view is unreliable, or a JSONL comparison record. This exists to keep the
user-facing demo grounded in detector measurements without network calls.
"""

import argparse
import json
import os
import re


BANNED_TERMS = [
    "injury", "injured", "injuries", "hurt", "hurts", "hurting",
    "damage", "damaged", "damaging", "risk", "risks", "risky",
    "diagnose", "diagnosed", "diagnosis", "diagnoses", "treat", "treated",
    "treating", "treatment", "treatments",
]

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_TOLERANCE_DEG = 0.05


def _number_claims(text: str) -> list[float]:
    return [float(match) for match in _NUMBER_RE.findall(text)]


def _contains_banned_term(text: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\w*\b", text, re.IGNORECASE) for term in BANNED_TERMS)


def _supported_numbers(detector_out: dict) -> list[float]:
    return [
        float(value)
        for value in detector_out["flags"].values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _template_bullets(detector_out: dict) -> list[str]:
    flags = detector_out["flags"]
    return [
        f"Knee depth: maximum knee flexion measured at {flags['knee_flexion_max_deg']:.2f} degrees.",
        f"Trunk lean: maximum trunk lean measured at {flags['trunk_lean_max_deg']:.2f} degrees.",
        f"Left-right symmetry: knee angle difference measured at {flags['symmetry_diff_deg']:.2f} degrees.",
    ]


def format_bullets(detector_out: dict) -> list[str]:
    """Return one refusal or exactly three fixed measurement sentences."""

    if detector_out["abstain_flag"] is True:
        return ["The view is unreliable, so no critique is given."]
    return _template_bullets(detector_out)


def verify_claims(bullets: list[str], detector_out: dict) -> list[str]:
    """Replace sentences containing unsupported numbers or terms with templates."""

    if detector_out["abstain_flag"] is True:
        return format_bullets(detector_out)
    templates = _template_bullets(detector_out)
    supported = _supported_numbers(detector_out)
    verified = []
    for index, bullet in enumerate(bullets):
        numbers = _number_claims(bullet)
        numbers_ok = bool(numbers) and all(
            any(abs(number - expected) <= _TOLERANCE_DEG for expected in supported)
            for number in numbers
        )
        if index >= len(templates) or not numbers_ok or _contains_banned_term(bullet):
            verified.append(templates[index] if index < len(templates) else templates[-1])
        else:
            verified.append(bullet)
    return verified


def run_demo(detector_out: dict) -> list[str]:
    """Format and verify the final sentences that the demo would print."""

    return verify_claims(format_bullets(detector_out), detector_out)


def _unsupported_number_count(texts: list[str] | str, detector_out: dict) -> int:
    text = texts if isinstance(texts, str) else " ".join(texts)
    supported = _supported_numbers(detector_out)
    return sum(
        not any(abs(number - expected) <= _TOLERANCE_DEG for expected in supported)
        for number in _number_claims(text)
    )


def log_hallucination_comparison(
    free_form_text: str,
    grounded_bullets: list[str],
    detector_out: dict,
    log_path: str = "results/halluc_log.jsonl",
) -> dict:
    """Append unsupported-number counts and supplied inputs to a JSONL log."""

    logged = {
        "free_form_unsupported_count": _unsupported_number_count(free_form_text, detector_out),
        "grounded_unsupported_count": _unsupported_number_count(grounded_bullets, detector_out),
        "free_form_text": free_form_text,
        "grounded_bullets": grounded_bullets,
        "detector_out": detector_out,
    }
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(logged) + "\n")
    return logged


def _sample_detector() -> dict:
    return {
        "flags": {
            "knee_flexion_max_deg": 82.4,
            "knee_depth_pass": False,
            "trunk_lean_max_deg": 18.2,
            "trunk_lean_pass": True,
            "symmetry_diff_deg": 7.1,
            "symmetry_pass": False,
        },
        "abstain_flag": False,
        "confidence": 0.91,
        "view_deg": 0.0,
    }


def _smoke_test() -> bool:
    detector = _sample_detector()
    refusal = format_bullets({**detector, "abstain_flag": True})
    bullets = run_demo(detector)
    fabricated = verify_claims(["Knee depth measured at 999.0 degrees.", *bullets[1:]], detector)
    planted = verify_claims(["Knee depth measured at 82.4 degrees; injury note.", *bullets[1:]], detector)
    return (
        len(refusal) == 1
        and "unreliable" in refusal[0].lower()
        and len(bullets) == 3
        and not _contains_banned_term(" ".join(bullets))
        and fabricated[0] == bullets[0]
        and planted[0] == bullets[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.demo:
        for bullet in run_demo(_sample_detector()):
            print(f"- {bullet}")
    elif args.smoke_test:
        print("PASS" if _smoke_test() else "FAIL")


if __name__ == "__main__":
    main()
