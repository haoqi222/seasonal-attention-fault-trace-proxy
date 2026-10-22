#!/usr/bin/env python3
"""Validate all 24 validation-only V4 blend preflight cells."""

from __future__ import annotations

import csv
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np


MODELS = ["unet_matched", "temporal_attn", "temporal_concat", "summer_only", "thermal_only", "optical_only"]
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranges(rows: np.ndarray) -> list[tuple[int, int]]:
    values = np.flatnonzero(rows)
    cuts = np.flatnonzero(np.diff(values) > 1)
    return list(zip(np.r_[values[0], values[cuts + 1]].tolist(), np.r_[values[cuts] + 1, values[-1] + 1].tolist()))


def line_values(array: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        valid = mask[:, 1:] & mask[:, :-1]
        delta = np.abs(array[:, 1:] - array[:, :-1])
        return np.array([np.mean(delta[:, i][valid[:, i]]) if valid[:, i].any() else np.nan for i in range(delta.shape[1])])
    valid = mask[1:] & mask[:-1]
    delta = np.abs(array[1:] - array[:-1])
    return np.array([np.mean(delta[i][valid[i]]) if valid[i].any() else np.nan for i in range(delta.shape[0])])


def seam_ratio(values: np.ndarray, boundaries: set[int]) -> float:
    seam = np.array([values[position - 1] for position in sorted(boundaries) if 0 < position <= len(values)])
    excluded = set()
    for position in boundaries:
        excluded.update((position - 2, position - 1, position))
    background = np.array([value for index, value in enumerate(values) if index not in excluded])
    seam = seam[np.isfinite(seam)]
    background = background[np.isfinite(background)]
    return float(np.mean(seam) / np.mean(background))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-file", type=Path, default=Path("03_configs/inference_v4_protocol.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol_file if args.protocol_file.is_absolute() else root / args.protocol_file
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    parent_sha = protocol["parent_v3"]["training_protocol_sha256"]
    preflight_root = root / "04_results/inference_v4_preflight" / protocol_sha
    parent_root = root / "01_data/interim/results_v3" / parent_sha
    checks = []
    comparisons = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    for model in MODELS:
        for fold in FOLDS:
            cell = preflight_root / model / fold / "seed_0"
            prefix = f"{model}:{fold}"
            check(prefix + ":directory", cell.is_dir())
            if not cell.is_dir():
                continue
            names = {path.name for path in cell.iterdir() if path.is_file()}
            expected = {"validation_probability.npy", "result.json", "environment.json", "config_snapshot.json", "run_manifest.json"}
            check(prefix + ":exact_file_set", names == expected, actual=sorted(names))
            check(prefix + ":no_test_probability", "test_probability.npy" not in names)
            manifest = json.loads((cell / "run_manifest.json").read_text(encoding="utf-8"))
            check(prefix + ":identity", manifest["model_id"] == model and manifest["fold_id"] == fold and
                  manifest["seed"] == 0 and manifest["preflight"] is True and manifest["roles"] == ["validation"])
            check(prefix + ":protocol", manifest["v4_protocol_sha256"] == protocol_sha)
            for name, digest in manifest["files"].items():
                check(prefix + f":hash:{name}", sha256(cell / name) == digest)
            snapshot = json.loads((cell / "config_snapshot.json").read_text(encoding="utf-8"))
            check(prefix + ":snapshot_exact", snapshot == protocol)
            result = json.loads((cell / "result.json").read_text(encoding="utf-8"))
            check(prefix + ":result_identity", result["model_id"] == model and result["fold_id"] == fold and result["seed"] == 0)
            fold_root = root / "01_data/interim/dataset_v3" / fold
            partition = np.load(fold_root / "partition.npy", mmap_mode="r")
            geographic = np.asarray(partition == 2)
            weighted = np.load(cell / "validation_probability.npy", mmap_mode="r")
            uniform = np.load(parent_root / model / fold / "seed_0" / "validation_probability.npy", mmap_mode="r")
            check(prefix + ":shape", weighted.shape == partition.shape)
            check(prefix + ":finite_range", np.isfinite(weighted).all() and float(weighted.min()) >= 0 and float(weighted.max()) <= 1)
            check(prefix + ":support", np.count_nonzero(weighted[~geographic]) == 0 and np.all(weighted[geographic] > 0))
            check(prefix + ":changed", not np.array_equal(weighted, uniform))
            row_ranges = ranges(np.any(geographic, axis=1))
            boundaries_by_axis = {
                "vertical": set(range(64, geographic.shape[1], 64)),
                "horizontal": set().union(*(set(range(start + 64, end, 64)) for start, end in row_ranges)),
            }
            for orientation, axis in (("vertical", 1), ("horizontal", 0)):
                old_ratio = seam_ratio(line_values(uniform, geographic, axis), boundaries_by_axis[orientation])
                new_ratio = seam_ratio(line_values(weighted, geographic, axis), boundaries_by_axis[orientation])
                relative = new_ratio / old_ratio
                comparisons.append({
                    "model_id": model, "fold_id": fold, "orientation": orientation,
                    "uniform_seam_ratio": old_ratio, "weighted_seam_ratio": new_ratio,
                    "weighted_over_uniform": relative,
                })

    maximum = float(max(row["weighted_over_uniform"] for row in comparisons))
    mean = float(np.mean([row["weighted_over_uniform"] for row in comparisons]))
    weighted_mean = float(np.mean([row["weighted_seam_ratio"] for row in comparisons]))
    acceptance_rule = protocol["preflight"]["acceptance"]
    check("preflight_cell_count", len(comparisons) == 48, comparisons=len(comparisons))
    if "material_uniform_seam_ratio_threshold" in acceptance_rule:
        material_threshold = float(acceptance_rule["material_uniform_seam_ratio_threshold"])
        material = [row for row in comparisons if row["uniform_seam_ratio"] >= material_threshold]
        relative_threshold = float(acceptance_rule["material_case_maximum_weighted_over_uniform"])
        absolute_threshold = float(acceptance_rule["all_case_maximum_weighted_seam_ratio"])
        mean_threshold = float(acceptance_rule["all_case_mean_weighted_seam_ratio"])
        check("material_cases_exist", len(material) > 0, count=len(material))
        check("material_cases_relative_reduction", all(row["weighted_over_uniform"] <= relative_threshold for row in material),
              maximum=max(row["weighted_over_uniform"] for row in material), threshold=relative_threshold)
        check("all_cases_absolute_seam_limit", all(row["weighted_seam_ratio"] <= absolute_threshold for row in comparisons),
              maximum=max(row["weighted_seam_ratio"] for row in comparisons), threshold=absolute_threshold)
        check("all_cases_mean_seam_limit", weighted_mean <= mean_threshold, mean=weighted_mean, threshold=mean_threshold)
        threshold_summary = {
            "material_uniform_seam_ratio_threshold": material_threshold,
            "material_case_maximum_weighted_over_uniform": relative_threshold,
            "all_case_maximum_weighted_seam_ratio": absolute_threshold,
            "all_case_mean_weighted_seam_ratio": mean_threshold,
        }
    else:
        threshold = float(acceptance_rule["weighted_over_uniform_mean_stride_jump_ratio_maximum"])
        check("every_stride_ratio_not_above_uniform", all(row["weighted_over_uniform"] <= 1 for row in comparisons), maximum=maximum)
        check("maximum_relative_stride_ratio_within_protocol", maximum <= threshold, maximum=maximum, threshold=threshold)
        threshold_summary = {"weighted_over_uniform_maximum": threshold}
    output = root / "04_results/inference_v4_preflight" / protocol_sha
    with (output / "preflight_stride_comparison_v4.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    failures = [entry for entry in checks if not entry["passed"]]
    acceptance = {
        "status": "accepted" if not failures else "rejected",
        "protocol_sha256": protocol_sha,
        "validation_only_cells": 24,
        "test_probability_files": 0,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "weighted_over_uniform_mean": mean,
        "weighted_over_uniform_maximum": maximum,
        "weighted_seam_ratio_mean": weighted_mean,
        "acceptance_thresholds": threshold_summary,
        "failures": failures,
    }
    (output / "stage11_preflight_acceptance_v4.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
