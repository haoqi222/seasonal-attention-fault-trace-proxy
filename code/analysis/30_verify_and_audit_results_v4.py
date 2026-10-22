#!/usr/bin/env python3
"""Independent V4.2 aggregation verification and formal test-map seam audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DEEP = ["unet_matched", "temporal_attn", "temporal_concat", "summer_only", "thermal_only", "optical_only"]
RF = ["rf_direct", "rf_context"]
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]
SEEDS = [0, 1, 2, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def ensemble(root: Path, model: str, fold: str, filename: str) -> np.ndarray:
    total = None
    for seed in SEEDS:
        value = np.load(root / model / fold / f"seed_{seed}" / filename, mmap_mode="r")
        if total is None:
            total = np.zeros(value.shape, dtype=np.float64)
        total += value
    return total / 4


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "03_configs/inference_v4_2_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    v4_sha = sha256(protocol_path)
    v3_sha = protocol["parent_v3"]["training_protocol_sha256"]
    v4_root = root / "01_data/interim/results_v4" / v4_sha
    v3_root = root / "01_data/interim/results_v3" / v3_sha
    aggregate = root / "04_results/experiment_v4_aggregate"
    seam_output = root / "04_results/prediction_seam_audit_v4"
    if seam_output.exists():
        raise FileExistsError(f"Refusing to overwrite {seam_output}")
    seam_output.mkdir(parents=True)
    checks = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    checksum_lines = (aggregate / "aggregate_checksums_v4.sha256").read_text(encoding="ascii").splitlines()
    check("aggregate_checksum_count", len(checksum_lines) == 11, count=len(checksum_lines))
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        check("aggregate_hash:" + name, sha256(aggregate / name) == digest)

    for model in DEEP:
        for fold in FOLDS:
            for seed in SEEDS:
                cell = v4_root / model / fold / f"seed_{seed}"
                manifest = json.loads((cell / "run_manifest.json").read_text(encoding="utf-8"))
                check(f"manifest_identity:{model}:{fold}:{seed}",
                      manifest["v4_protocol_sha256"] == v4_sha and manifest["model_id"] == model and
                      manifest["fold_id"] == fold and manifest["seed"] == seed and not manifest["preflight"])
                for name, digest in manifest["files"].items():
                    check(f"run_hash:{model}:{fold}:{seed}:{name}", sha256(cell / name) == digest)

    fold_table = {(row["model_id"], row["fold_id"]): row for row in csv_rows(aggregate / "fold_seed_ensemble_metrics_v4.csv")}
    summary_table = {row["model_id"]: row for row in csv_rows(aggregate / "model_summary_v4.csv")}
    current_arrays = {}
    seam_rows = []
    for model in [*DEEP, *RF]:
        source = v4_root if model in DEEP else v3_root
        fold_ap = []
        fold_roc = []
        for fold in FOLDS:
            folder = root / "01_data/interim/dataset_v3" / fold
            y = np.load(folder / "y.npy", mmap_mode="r")
            test = np.load(folder / "test.npy", mmap_mode="r").astype(bool)
            partition = np.load(folder / "partition.npy", mmap_mode="r")
            probability = ensemble(source, model, fold, "test_probability.npy")
            selected = test & (y != 2)
            truth = (y[selected] == 1).astype(np.int8)
            score = probability[selected]
            ap = float(average_precision_score(truth, score))
            roc = float(roc_auc_score(truth, score))
            fold_ap.append(ap)
            fold_roc.append(roc)
            stored = fold_table[(model, fold)]
            check(f"ensemble_ap:{model}:{fold}", abs(ap - float(stored["test_average_precision"])) <= 1e-12)
            check(f"ensemble_roc:{model}:{fold}", abs(roc - float(stored["test_roc_auc"])) <= 1e-12)
            current_arrays[(model, fold)] = probability
            if model in DEEP:
                old_probability = ensemble(v3_root, model, fold, "test_probability.npy")
                geographic = np.asarray(partition == 3)
                row_ranges = ranges(np.any(geographic, axis=1))
                boundary_sets = {
                    "vertical": set(range(64, geographic.shape[1], 64)),
                    "horizontal": set().union(*(set(range(start + 64, end, 64)) for start, end in row_ranges)),
                }
                for orientation, axis in (("vertical", 1), ("horizontal", 0)):
                    old_ratio = seam_ratio(line_values(old_probability, geographic, axis), boundary_sets[orientation])
                    new_ratio = seam_ratio(line_values(probability, geographic, axis), boundary_sets[orientation])
                    seam_rows.append({"model_id": model, "fold_id": fold, "orientation": orientation,
                                      "v3_uniform_seam_ratio": old_ratio, "v4_weighted_seam_ratio": new_ratio,
                                      "v4_over_v3": new_ratio / old_ratio})
        summary = summary_table[model]
        check(f"summary_ap_mean:{model}", abs(np.mean(fold_ap) - float(summary["fold_ensemble_average_precision_mean"])) <= 1e-12)
        check(f"summary_ap_sd:{model}", abs(np.std(fold_ap, ddof=1) - float(summary["fold_ensemble_average_precision_sd"])) <= 1e-12)
        check(f"summary_roc_mean:{model}", abs(np.mean(fold_roc) - float(summary["fold_ensemble_roc_auc_mean"])) <= 1e-12)

    pairwise_table = {row["comparison_id"]: row for row in csv_rows(aggregate / "pairwise_summary_v4.csv")}
    comparisons = {
        "temporal_attn_vs_unet": ("temporal_attn", "unet_matched"),
        "temporal_attn_vs_concat": ("temporal_attn", "temporal_concat"),
        "temporal_attn_vs_summer": ("temporal_attn", "summer_only"),
        "temporal_attn_vs_thermal": ("temporal_attn", "thermal_only"),
        "temporal_attn_vs_optical": ("temporal_attn", "optical_only"),
        "rf_context_vs_direct": ("rf_context", "rf_direct"),
    }
    for name, (left, right) in comparisons.items():
        differences = np.array([float(fold_table[(left, fold)]["test_average_precision"]) -
                                float(fold_table[(right, fold)]["test_average_precision"]) for fold in FOLDS])
        stored = pairwise_table[name]
        check("pairwise_mean:" + name, abs(differences.mean() - float(stored["mean_ap_difference_left_minus_right"])) <= 1e-12)
        check("pairwise_sd:" + name, abs(differences.std(ddof=1) - float(stored["sd_ap_difference"])) <= 1e-12)
        check("pairwise_wins:" + name, int((differences > 0).sum()) == int(stored["left_wins"]) and
              int((differences < 0).sum()) == int(stored["right_wins"]))

    with (seam_output / "v3_v4_test_stride_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seam_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(seam_rows)
    new_ratios = np.array([row["v4_weighted_seam_ratio"] for row in seam_rows])
    old_ratios = np.array([row["v3_uniform_seam_ratio"] for row in seam_rows])
    severe = [row for row in seam_rows if row["v3_uniform_seam_ratio"] >= 2.0]
    check("formal_test_mean_seam_ratio", float(new_ratios.mean()) <= 1.2, mean=float(new_ratios.mean()))
    check("formal_test_max_seam_ratio", float(new_ratios.max()) <= 1.5, maximum=float(new_ratios.max()))
    check("formal_test_severe_cases_halved", all(row["v4_over_v3"] <= 0.5 for row in severe),
          count=len(severe), maximum=max(row["v4_over_v3"] for row in severe))
    seam_report = {
        "status": "accepted" if checks[-3]["passed"] and checks[-2]["passed"] and checks[-1]["passed"] else "rejected",
        "deep_model_fold_orientation_records": len(seam_rows),
        "v3_uniform_mean_seam_ratio": float(old_ratios.mean()),
        "v4_weighted_mean_seam_ratio": float(new_ratios.mean()),
        "v4_weighted_max_seam_ratio": float(new_ratios.max()),
        "severe_v3_records": len(severe),
        "severe_v4_over_v3_maximum": float(max(row["v4_over_v3"] for row in severe)),
    }
    (seam_output / "test_seam_acceptance_v4.json").write_text(json.dumps(seam_report, indent=2) + "\n", encoding="utf-8")

    failures = [row for row in checks if not row["passed"]]
    report = {
        "status": "accepted" if not failures else "rejected",
        "checks_passed": len(checks) - len(failures), "checks_total": len(checks),
        "failures": failures, "test_seam_audit": seam_report,
    }
    (aggregate / "independent_verification_v4.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"],
                      "checks_total": report["checks_total"], "test_seam_audit": seam_report}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
