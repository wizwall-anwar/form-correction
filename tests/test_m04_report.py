"""Test report packaging, synthetic stamps, figures, and result bundles."""

import json
import zipfile

from src.m04_report import (build_results_bundle, build_table2, build_table4,
                            make_risk_coverage_figure, stamp_synthetic)


def test_stamp_synthetic_marks_only_synthetic_tables():
    table = {"rows": []}
    assert stamp_synthetic(table, True)["data_quality"] == "SYNTHETIC - not evidence"
    assert stamp_synthetic(table, False) == table
    assert "data_quality" not in table


def test_table2_marks_nonzero_angles_as_synthetic():
    table = build_table2([{"angle": angle} for angle in (0, 15, 30, 45, 60)])
    assert [row["is_synthetic_view"] for row in table["rows"]] == [False, True, True, True, True]


def test_table4_uses_reliability_label():
    table = build_table4({"0": {}, "45": {}}, {"free_form": 0.1})
    assert "validity" not in json.dumps(table).lower()
    assert "model-vs-clinician reliability" in table["comment"]


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
