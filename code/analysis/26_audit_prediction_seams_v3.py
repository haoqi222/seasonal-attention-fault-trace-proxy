#!/usr/bin/env python3
"""Quantify whether formal V3 score maps contain stride-aligned seams."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROTOCOL_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]
MODELS = ["temporal_attn", "temporal_concat", "unet_matched", "rf_direct"]


def ensemble(result_root: Path, model: str, fold: str) -> np.ndarray:
    return np.mean(np.stack([
        np.load(result_root / model / fold / f"seed_{seed}" / "test_probability.npy", mmap_mode="r")
        for seed in range(4)
    ]), axis=0, dtype=np.float64)


def contiguous_ranges(rows: np.ndarray) -> list[tuple[int, int]]:
    values = np.flatnonzero(rows)
    cuts = np.flatnonzero(np.diff(values) > 1)
    starts = np.r_[values[0], values[cuts + 1]]
    ends = np.r_[values[cuts] + 1, values[-1] + 1]
    return list(zip(starts.tolist(), ends.tolist()))


def stride_lines(start: int, end: int, stride: int = 64) -> set[int]:
    return set(range(start + stride, end, stride))


def line_scores(array: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        valid = mask[:, 1:] & mask[:, :-1]
        delta = np.abs(array[:, 1:] - array[:, :-1])
        return np.array([float(np.mean(delta[:, i][valid[:, i]])) if valid[:, i].any() else np.nan
                         for i in range(delta.shape[1])])
    valid = mask[1:, :] & mask[:-1, :]
    delta = np.abs(array[1:, :] - array[:-1, :])
    return np.array([float(np.mean(delta[i, :][valid[i, :]])) if valid[i, :].any() else np.nan
                     for i in range(delta.shape[0])])


def summarize(lines: np.ndarray, seam_indices: set[int]) -> dict:
    # Difference index i is the boundary between pixels i and i+1, hence line c uses i=c-1.
    seam = np.array([lines[c - 1] for c in sorted(seam_indices) if 0 < c <= len(lines)], dtype=float)
    exclusion = set()
    for c in seam_indices:
        exclusion.update({c - 2, c - 1, c})
    background = np.array([v for i, v in enumerate(lines) if i not in exclusion], dtype=float)
    seam = seam[np.isfinite(seam)]
    background = background[np.isfinite(background)]
    return {
        "seam_line_count": int(len(seam)),
        "seam_mean_abs_jump": float(np.mean(seam)),
        "background_mean_abs_jump": float(np.mean(background)),
        "seam_to_background_ratio": float(np.mean(seam) / np.mean(background)),
        "seam_median_abs_jump": float(np.median(seam)),
        "background_median_abs_jump": float(np.median(background)),
        "seam_to_background_median_ratio": float(np.median(seam) / np.median(background)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "04_results/prediction_seam_audit_v3").resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_root = root / "01_data/interim/results_v3" / PROTOCOL_SHA
    rows = []
    for fold in FOLDS:
        fold_root = root / "01_data/interim/dataset_v3" / fold
        partition = np.load(fold_root / "partition.npy", mmap_mode="r")
        y = np.load(fold_root / "y.npy", mmap_mode="r")
        geographic = np.asarray(partition == 3)
        eligible = geographic & (y != 2)
        row_ranges = contiguous_ranges(np.any(geographic, axis=1))
        y_seams = set().union(*(stride_lines(start, end) for start, end in row_ranges))
        x_seams = stride_lines(0, geographic.shape[1])
        for model in MODELS:
            score = ensemble(result_root, model, fold)
            mask = eligible if model == "rf_direct" else geographic
            for axis_name, axis, seams in (("vertical", 1, x_seams), ("horizontal", 0, y_seams)):
                stats = summarize(line_scores(score, mask, axis), seams)
                rows.append({"model_id": model, "fold_id": fold, "boundary_orientation": axis_name, **stats})
    csv_path = output / "stride_boundary_jump_audit_v3.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    deep = [r for r in rows if r["model_id"] != "rf_direct"]
    rf = [r for r in rows if r["model_id"] == "rf_direct"]
    report = {
        "status": "diagnostic_only",
        "protocol_sha256": PROTOCOL_SHA,
        "definition": "mean absolute adjacent-pixel score jump on nominal 64-pixel stride lines divided by other lines",
        "deep_mean_ratio": float(np.mean([r["seam_to_background_ratio"] for r in deep])),
        "deep_max_ratio": float(np.max([r["seam_to_background_ratio"] for r in deep])),
        "rf_direct_mean_ratio": float(np.mean([r["seam_to_background_ratio"] for r in rf])),
        "rf_direct_max_ratio": float(np.max([r["seam_to_background_ratio"] for r in rf])),
        "note": "This diagnostic identifies stride alignment but does not by itself establish the causal source of visible map structure.",
    }
    (output / "prediction_seam_audit_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
