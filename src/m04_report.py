"""Package computed detector and agreement results into report artifacts.

Inputs are plain dictionaries/lists from the other pipeline modules. Outputs
are table dictionaries, a small matplotlib figure, a merged manifest, and a
zip archive; this exists to format results without rerunning any computation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_table1(data_source: str, n_subjects: int, n_sequences: int,
                subject_split_acc: float, clip_split_acc: float) -> dict:
    """Build the dataset and split-accuracy summary table."""
    return {
        "data_source": data_source,
        "n_subjects": n_subjects,
        "n_sequences": n_sequences,
        "subject_split_accuracy": subject_split_acc,
        "clip_split_accuracy": clip_split_acc,
        "note": "The clip split is intentionally leaky and shown only to illustrate the gap.",
    }


def build_table2(rows: list[dict]) -> dict:
    """Build the per-angle detector and flag-error table."""
    marked_rows = []
    for row in rows:
        marked = dict(row)
        marked["is_synthetic_view"] = marked.get("angle", marked.get("view_deg")) != 0
        marked_rows.append(marked)
    return {
        "rows": marked_rows,
        "caption": "Detector performance and flag error for SYNTHETIC rotated views, not real camera views.",
        "metadata": {"views": "SYNTHETIC rotated views, not real camera views"},
    }


def build_table3(ece: float, precision_at_coverage: dict, selective_risk: dict) -> dict:
    """Build the calibration and abstention summary from supplied metrics."""
    return {"ece": ece, "precision_at_coverage": precision_at_coverage, "selective_risk": selective_risk}


def build_table4(kappa_report: dict, hallucination_rates: dict) -> dict:
    """Build model-vs-clinician reliability and hallucination summaries."""
    return {
        "kappa_report": kappa_report,
        "kappa_0_vs_45": kappa_report,
        "hallucination_rates": hallucination_rates,
        "comment": "Labeled model-vs-clinician reliability.",
    }


def make_risk_coverage_figure(coverage: list[float], risk: list[float],
                             out_path: str = "results/fig1_risk_coverage.png") -> str:
    """Plot supplied selective risk over coverage and return the output path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot(coverage, risk, marker="o")
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Selective risk")
    axis.set_title("Selective risk vs coverage")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return str(path)


def stamp_synthetic(table: dict, is_synthetic: bool) -> dict:
    """Mark a table as synthetic-only while leaving real tables untouched."""
    if is_synthetic:
        stamped = dict(table)
        stamped["data_quality"] = "SYNTHETIC - not evidence"
        return stamped
    return table


def save_manifest(data_source: str, seed: int,
                  out_path: str = "results/manifest.json") -> str:
    """Merge source and seed into a manifest without removing existing fields."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update({"source": data_source, "seed": seed})
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path)


def build_results_bundle(out_dir: str = "results",
                         zip_path: str = "results/results_bundle.zip") -> str:
    """Zip all current result files, including an optional weights directory."""
    source_dir = Path(out_dir)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        if source_dir.exists():
            for path in source_dir.rglob("*"):
                if path.is_file() and path.resolve() != destination.resolve():
                    archive.write(path, path.relative_to(source_dir).as_posix())
    return str(destination)


def _smoke_test() -> bool:
    """Exercise report formatting and file outputs with small offline inputs."""
    with tempfile.TemporaryDirectory() as temporary:
        results = Path(temporary)
        build_table1("synthetic", 2, 4, 0.5, 0.9)
        build_table2([{"angle": angle, "accuracy": 0.5} for angle in (0, 15, 30, 45, 60)])
        build_table3(0.1, {0.5: 0.8}, {"coverage": [0.5], "risk": [0.2]})
        build_table4({"0": {"cohen_kappa": 0.1}, "45": {"cohen_kappa": 0.2}}, {"free_form": 0.1})
        figure = make_risk_coverage_figure([0.5, 1.0], [0.2, 0.3], str(results / "figure.png"))
        assert Path(figure).is_file()
        assert stamp_synthetic({}, True)["data_quality"] == "SYNTHETIC - not evidence"
        save_manifest("synthetic", 1, str(results / "manifest.json"))
        bundle = build_results_bundle(str(results), str(results / "results_bundle.zip"))
        assert Path(bundle).is_file() and Path(bundle).stat().st_size > 0
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        try:
            print("PASS" if _smoke_test() else "FAIL")
        except Exception as error:
            print(f"FAIL: {error}")
