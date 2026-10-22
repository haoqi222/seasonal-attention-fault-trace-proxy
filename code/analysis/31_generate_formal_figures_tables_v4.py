#!/usr/bin/env python3
"""Generate the V4.2 figure and table package without modifying the V3 package."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "05_figures_tables/formal_v4"
V4_SHA = "43315ba42f2686b339644e0e89afd964bab32ef4647aa9189bfd4eea96118b1b"
V3_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
DEEP = {"unet_matched", "temporal_attn", "temporal_concat", "summer_only", "thermal_only", "optical_only"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source_path = ROOT / "02_scripts/analysis/25_generate_formal_figures_tables_v3.py"
    spec = importlib.util.spec_from_file_location("figures_v3", source_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    module.PROTOCOL_SHA = V4_SHA

    def load_v4_ensemble(_result_root: Path, model: str, fold: str) -> np.ndarray:
        source = (ROOT / "01_data/interim/results_v4" / V4_SHA if model in DEEP else
                  ROOT / "01_data/interim/results_v3" / V3_SHA)
        arrays = [np.load(source / model / fold / f"seed_{seed}" / "test_probability.npy", mmap_mode="r")
                  for seed in range(4)]
        return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)

    module.load_ensemble = load_v4_ensemble
    font_name = module.configure_font(Path("/tmp/times"))
    aggregate = ROOT / "04_results/experiment_v4_aggregate"
    fold_rows = rows(aggregate / "fold_seed_ensemble_metrics_v4.csv")
    summary_rows = rows(aggregate / "model_summary_v4.csv")
    pairwise_rows = rows(aggregate / "pairwise_summary_v4.csv")
    training_rows = rows(ROOT / "04_results/experiment_v3_aggregate/training_diagnostics_v3.csv")
    v3_protocol = json.loads((ROOT / "03_configs/experiment_v3_protocol.json").read_text(encoding="utf-8"))
    v4_result_root = ROOT / "01_data/interim/results_v4" / V4_SHA

    module.figure_1_spatial_cv(ROOT, OUTPUT)
    module.figure_2_workflow(OUTPUT)
    module.figure_3_fold_performance(fold_rows, summary_rows, OUTPUT)
    module.figure_4_pr_curves(ROOT, v4_result_root, OUTPUT)
    module.figure_5_maps(ROOT, v4_result_root, OUTPUT)
    module.figure_6_training(training_rows, OUTPUT)
    module.make_tables(ROOT, OUTPUT, v3_protocol, fold_rows, summary_rows, pairwise_rows)
    for path in list(OUTPUT.iterdir()):
        if path.is_file() and "_v3_" in path.name:
            path.rename(path.with_name(path.name.replace("_v3_", "_v4_")))

    figures = sorted(OUTPUT.glob("fig*.png"))
    image_records = []
    for path in figures:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            image_records.append({"file": path.name, "width": image.width, "height": image.height,
                                  "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1])})
    seam = json.loads((ROOT / "04_results/prediction_seam_audit_v4/test_seam_acceptance_v4.json").read_text(encoding="utf-8"))
    qa = {
        "status": "provisional_pending_seam_exception_review",
        "v4_protocol_sha256": V4_SHA,
        "font": font_name,
        "configured_minimum_visible_font_pt": 14,
        "unique_figure_filenames": len({path.name for path in figures}) == len(figures),
        "all_png_approximately_300_dpi": all(abs(r["dpi_x"] - 300) < 1 and abs(r["dpi_y"] - 300) < 1 for r in image_records),
        "test_seam_audit": seam,
        "accepted_for_release": [
            "fig1_v4_spatial_cross_validation_design.png",
            "fig2_v4_frozen_evaluation_workflow.png",
            "fig3_v4_spatial_fold_performance.png",
            "fig4_v4_seed_ensemble_pr_curves.png"
        ],
        "pending_manual_review": ["fig5_v4_out_of_fold_percentile_maps.png"],
        "supplementary_candidate": ["fig6_v4_training_diagnostics.png"],
        "images": image_records,
    }
    (OUTPUT / "figure_qa_v4.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    readme = (
        "# V4.2 candidate formal figures and tables\n\n"
        "Deep-model results use center-weighted V4.2 re-inference from frozen V3 checkpoints; RF results are exact V3 reuse.\n\n"
        "All figure filenames are unique, panel labels use lowercase parenthetical form, visible text is configured at 14 pt or larger, and PNG export is approximately 300 ppi.\n\n"
        "Figure 5 remains provisional because the independent test audit passed both absolute seam limits but missed one severe-case relative-reduction threshold by 0.00493. It must be visually reviewed and the exception disclosed before release insertion.\n"
    )
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    manifest = {"status": qa["status"], "v4_protocol_sha256": V4_SHA,
                "figure_count": len(figures), "table_count": len(list(OUTPUT.glob("table*.csv"))),
                "files": sorted(path.name for path in OUTPUT.iterdir() if path.is_file())}
    (OUTPUT / "formal_v4_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    names = sorted(path.name for path in OUTPUT.iterdir() if path.is_file())
    (OUTPUT / "formal_v4_checksums.sha256").write_text(
        "".join(f"{sha256(OUTPUT / name)}  {name}\n" for name in names), encoding="ascii", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
