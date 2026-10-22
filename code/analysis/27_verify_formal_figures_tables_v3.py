#!/usr/bin/env python3
"""Independent structural and numeric verification of V3 figure/table exports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "05_figures_tables/formal_v3"
    aggregate = root / "04_results/experiment_v3_aggregate"
    seam = root / "04_results/prediction_seam_audit_v3/prediction_seam_audit_v3.json"
    checks: list[dict] = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    expected_figures = [
        "fig1_v3_spatial_cross_validation_design.png",
        "fig2_v3_frozen_evaluation_workflow.png",
        "fig3_v3_spatial_fold_performance.png",
        "fig4_v3_seed_ensemble_pr_curves.png",
        "fig5_v3_out_of_fold_percentile_maps.png",
        "fig6_v3_training_diagnostics.png",
    ]
    expected_tables = [
        "table1_v3_spatial_fold_test_sets.csv",
        "table2_v3_model_definitions.csv",
        "table3_v3_model_performance.csv",
        "table4_v3_pairwise_fold_ap_differences.csv",
    ]
    actual_figures = sorted(p.name for p in folder.glob("fig*.png"))
    actual_tables = sorted(p.name for p in folder.glob("table*.csv"))
    check("six_expected_unique_figure_names", actual_figures == sorted(expected_figures), actual=actual_figures)
    check("four_expected_unique_table_names", actual_tables == sorted(expected_tables), actual=actual_tables)

    for name in expected_figures:
        with Image.open(folder / name) as image:
            dpi = image.info.get("dpi", (0, 0))
            check(f"png_dimensions_positive:{name}", image.width > 1000 and image.height > 1000,
                  width=image.width, height=image.height)
            check(f"png_300_dpi:{name}", abs(float(dpi[0]) - 300) < 1 and abs(float(dpi[1]) - 300) < 1,
                  dpi_x=dpi[0], dpi_y=dpi[1])

    checksum_map = {}
    for line in (folder / "formal_v3_checksums.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksum_map[name] = digest
    for name, digest in checksum_map.items():
        check(f"checksum:{name}", sha256(folder / name) == digest)
    check("checksum_entry_count", len(checksum_map) == 13, count=len(checksum_map))

    expected_counts = {
        expected_tables[0]: 4,
        expected_tables[1]: 8,
        expected_tables[2]: 8,
        expected_tables[3]: 6,
    }
    for name, count in expected_counts.items():
        check(f"table_row_count:{name}", len(rows(folder / name)) == count, count=len(rows(folder / name)))

    exported = {r["model"]: r for r in rows(folder / expected_tables[2])}
    source = {r["model_id"]: r for r in rows(aggregate / "model_summary_v3.csv")}
    labels = {
        "temporal_attn": "Temporal attention", "temporal_concat": "Seasonal concatenation",
        "unet_matched": "U-Net matched", "summer_only": "Summer-only",
        "optical_only": "Optical-only", "rf_context": "RF context",
        "rf_direct": "RF direct", "thermal_only": "Thermal-only",
    }
    for model_id, label in labels.items():
        difference = abs(float(exported[label]["average_precision_mean"]) -
                         float(source[model_id]["fold_ensemble_average_precision_mean"]))
        check(f"table3_ap_matches_rounded_source:{model_id}", difference <= 0.00005, difference=difference)

    qa = json.loads((folder / "figure_qa_v3.json").read_text(encoding="utf-8"))
    seam_report = json.loads(seam.read_text(encoding="utf-8"))
    check("qa_status_is_provisional", qa["status"] == "provisional_not_submission_ready")
    check("figure5_rejected", expected_figures[4] in qa["rejected_pending_inference_correction"])
    check("figure5_not_accepted", expected_figures[4] not in qa["accepted_for_release"])
    check("minimum_font_14_pt", qa["configured_minimum_visible_font_pt"] >= 14)
    check("deep_stride_signal_material", seam_report["deep_mean_ratio"] > 2.0,
          ratio=seam_report["deep_mean_ratio"])
    check("rf_stride_control_near_one", seam_report["rf_direct_mean_ratio"] < 1.2,
          ratio=seam_report["rf_direct_mean_ratio"])

    failures = [entry for entry in checks if not entry["passed"]]
    report = {
        "status": "passed_with_declared_submission_blocker" if not failures else "failed",
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "failures": failures,
        "submission_blocker": qa["blocking_reason"],
        "checks": checks,
    }
    destination = folder / "independent_figure_table_verification_v3.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checks_passed", "checks_total", "submission_blocker")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
