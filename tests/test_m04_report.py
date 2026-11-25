"""Test report packaging, synthetic stamps, figures, and result bundles."""

import json
import zipfile

from src.m04_report import (build_results_bundle, build_table1, build_table2, build_table3, build_table4,
                            make_risk_coverage_figure, stamp_synthetic)


def test_stamp_synthetic_marks_only_synthetic_tables():
    table = {"rows": []}
    assert stamp_synthetic(table, True)["data_quality"] == "SYNTHETIC - not evidence"
    assert stamp_synthetic(table, False) == table
    assert "data_quality" not in table


def test_table2_marks_nonzero_angles_as_synthetic():
    table = build_table2([{"angle": angle, "spearman_rho": 0.5, "mae": 0.1, "flag_error": 1.0}
                          for angle in (0, 15, 30, 45, 60)])
    assert [row["is_synthetic_view"] for row in table["rows"]] == [False, True, True, True, True]


def test_table1_reports_regression_and_single_sequence_limit() -> None:
    subject = {"spearman_rho": 0.6, "mae": 0.1, "spearman_rho_ci95": [0.2, 0.8],
               "mae_ci95": [0.05, 0.2], "mae_mean_baseline": 0.12, "n_test": 15}
    clip = {"spearman_rho": 0.6, "mae": 0.1, "n_test": 15}

    table = build_table1("manual", 75, 75, subject, clip, 1)

    assert table["subject_split"] == subject
    assert table["clip_split"] == clip
    assert "held-out subjects" in table["note"]
    assert "Kinect-derived" in table["note"]
    assert "phone transfer untested" in table["note"]
    assert "flags are measurements" in table["note"]
    assert "not informative" in table["note"]


def test_table3_labels_selective_accuracy_and_test_context() -> None:
    table = build_table3(0.1, {"0.5": 0.8}, {"coverage": [0.5], "risk": [0.2]}, 15, {"0": 5})

    assert table["selective_accuracy_at_coverage"] == {"0.5": 0.8}
    assert table["n_test"] == 15
    assert table["test_class_counts"] == {"0": 5}
    assert "precision_at_coverage" not in table


def test_table4_uses_reliability_label():
    table = build_table4({"0": {}, "45": {}}, {"free_form": 0.1})
    assert "validity" not in json.dumps(table).lower()
    assert "model-vs-clinician reliability" in table["comment"]


def test_table1_carries_per_flag_subject_split_when_supplied() -> None:
    """When a per-flag correlations dict is supplied, table1 surfaces it as-is."""
    subject = {"spearman_rho": -0.5, "mae": 0.08, "spearman_rho_ci95": [-0.8, -0.1],
               "mae_ci95": [0.05, 0.1], "mae_mean_baseline": 0.07, "n_test": 23}
    clip = {"spearman_rho": -0.3, "mae": 0.08, "n_test": 23}
    per_flag = {"knee_flexion_max_deg": -0.2, "trunk_lean_max_deg": 0.1,
                "symmetry_diff_deg": -0.15}

    table = build_table1("manual", 75, 75, subject, clip, 1,
                         per_flag_subject_split=per_flag)

    assert table["per_flag_subject_split"] == per_flag


def test_table1_omits_per_flag_subject_split_when_not_supplied() -> None:
    """Backward-compatible: omitting the kwarg leaves the field absent."""
    subject = {"spearman_rho": 0.0, "mae": 0.0, "spearman_rho_ci95": [0.0, 0.0],
               "mae_ci95": [0.0, 0.0], "mae_mean_baseline": 0.0, "n_test": 1}
    clip = {"spearman_rho": 0.0, "mae": 0.0, "n_test": 1}
    table = build_table1("manual", 1, 1, subject, clip, 1)
    assert "per_flag_subject_split" not in table


def test_table4_carries_detector_vs_clinician_when_supplied() -> None:
    """When a detector-vs-clinician dict is supplied, table4 surfaces it."""
    detector = {"n_scored": 55, "cohen_kappa": -0.10,
                "quadratic_weighted_cohen_kappa": -0.18}
    table = build_table4({"0": {}, "45": {}}, {"free_form": 0.0},
                         detector_vs_clinician=detector)
    assert table["detector_vs_clinician_kappa"] == detector


def test_table4_omits_detector_vs_clinician_when_not_supplied() -> None:
    table = build_table4({"0": {}, "45": {}}, {"free_form": 0.0})
    assert "detector_vs_clinician_kappa" not in table


def test_risk_coverage_figure_writes_file(tmp_path):
    output = tmp_path / "figure.png"
    path = make_risk_coverage_figure([0.5, 1.0], [0.1, 0.2], str(output))
    assert path == str(output)
    assert output.is_file() and output.stat().st_size > 0


def test_results_bundle_contains_expected_members(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "table1.json").write_text("{}", encoding="utf-8")
    (results / "manifest.json").write_text("{}", encoding="utf-8")
    (results / "figure.png").write_bytes(b"png")
    weights = results / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"weights")
    bundle = tmp_path / "bundle.zip"
    path = build_results_bundle(str(results), str(bundle))
    assert bundle.is_file() and bundle.stat().st_size > 0
    assert path == str(bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"table1.json", "manifest.json", "figure.png", "weights/model.bin"}
