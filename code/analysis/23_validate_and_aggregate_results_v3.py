#!/usr/bin/env python3
"""Validate all 128 V3 runs and create protocol-defined aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROTOCOL_REL = Path("03_configs/experiment_v3_protocol.json")
LOCK_REL = Path("04_results/training_protocol_v3/protocol_lock_v3.json")
EXECUTION_SUMMARY_REL = Path("99_logs/experiment_v3")
AUDIT_REL = Path("04_results/experiment_v3_aggregate")
METRIC_KEYS = [
    "n",
    "positive",
    "positive_rate",
    "average_precision",
    "roc_auc",
    "precision_top_1pct",
    "recall_top_1pct",
    "precision_top_5pct",
    "recall_top_5pct",
    "threshold",
    "f1_at_0_5",
    "precision_at_0_5",
    "recall_at_0_5",
    "tp_at_0_5",
    "fp_at_0_5",
    "fn_at_0_5",
]
SUMMARY_METRICS = [
    "average_precision",
    "roc_auc",
    "precision_top_1pct",
    "recall_top_1pct",
    "precision_top_5pct",
    "recall_top_5pct",
    "f1_at_0_5",
    "precision_at_0_5",
    "recall_at_0_5",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(np.asarray(probability[selected], dtype=np.float64), 0.0, 1.0)
    result = {
        "n": int(len(truth)),
        "positive": int(truth.sum()),
        "positive_rate": float(truth.mean()),
        "average_precision": float(average_precision_score(truth, score)),
        "roc_auc": float(roc_auc_score(truth, score)),
    }
    for fraction in (0.01, 0.05):
        count = max(1, int(math.ceil(len(score) * fraction)))
        top = np.argpartition(score, -count)[-count:]
        true_positive = int(truth[top].sum())
        label = f"top_{int(fraction * 100)}pct"
        result[f"precision_{label}"] = true_positive / count
        result[f"recall_{label}"] = true_positive / max(int(truth.sum()), 1)
    predicted = score >= 0.5
    tp = int((predicted & (truth == 1)).sum())
    fp = int((predicted & (truth == 0)).sum())
    fn = int((~predicted & (truth == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result.update(
        {
            "threshold": 0.5,
            "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
            "precision_at_0_5": precision,
            "recall_at_0_5": recall,
            "tp_at_0_5": tp,
            "fp_at_0_5": fp,
            "fn_at_0_5": fn,
        }
    )
    return result


def metric_delta(stored: dict, recomputed: dict) -> tuple[float, list[str]]:
    maximum = 0.0
    errors: list[str] = []
    integer_keys = {"n", "positive", "tp_at_0_5", "fp_at_0_5", "fn_at_0_5"}
    for key in METRIC_KEYS:
        if key not in stored:
            errors.append(f"missing_metric:{key}")
            continue
        if key in integer_keys:
            if int(stored[key]) != int(recomputed[key]):
                errors.append(f"metric_mismatch:{key}:{stored[key]}:{recomputed[key]}")
        else:
            delta = abs(float(stored[key]) - float(recomputed[key]))
            maximum = max(maximum, delta)
            if not math.isfinite(delta) or delta > 5e-6:
                errors.append(f"metric_mismatch:{key}:delta={delta}")
    return maximum, errors


def expected_cells(protocol: dict) -> list[dict]:
    rows: list[dict] = []
    ordinal = 0
    for model in protocol["deep_models"]:
        for fold_id in protocol["data"]["folds"]:
            for seed in protocol["seeds"]:
                ordinal += 1
                rows.append(
                    {
                        "ordinal": ordinal,
                        "model_id": model["model_id"],
                        "model_kind": "deep",
                        "analysis_tier": model["analysis_tier"],
                        "expected_parameters": model["expected_trainable_parameters"],
                        "fold_id": fold_id,
                        "seed": int(seed),
                    }
                )
    for model in protocol["random_forest_models"]:
        for fold_id in protocol["data"]["folds"]:
            for seed in protocol["seeds"]:
                ordinal += 1
                rows.append(
                    {
                        "ordinal": ordinal,
                        "model_id": model["model_id"],
                        "model_kind": "random_forest",
                        "analysis_tier": model["analysis_tier"],
                        "expected_parameters": "",
                        "fold_id": fold_id,
                        "seed": int(seed),
                    }
                )
    if len(rows) != 128:
        raise RuntimeError(len(rows))
    return rows


def load_fold_arrays(root: Path, fold_id: str) -> dict:
    fold_root = root / "01_data/interim/dataset_v3" / fold_id
    arrays = {
        "y": np.load(fold_root / "y.npy", mmap_mode="r"),
        "partition": np.load(fold_root / "partition.npy", mmap_mode="r"),
        "validation": np.load(fold_root / "val.npy", mmap_mode="r").astype(bool),
        "test": np.load(fold_root / "test.npy", mmap_mode="r").astype(bool),
    }
    if any(
        arrays["y"].shape != arrays[name].shape
        for name in ("partition", "validation", "test")
    ):
        raise RuntimeError(f"Fold array shape mismatch: {fold_id}")
    return arrays


def validate_run(
    root: Path,
    result_root: Path,
    protocol: dict,
    protocol_sha: str,
    cell: dict,
    fold_arrays: dict,
) -> tuple[dict, dict, dict]:
    run_dir = result_root / cell["model_id"] / cell["fold_id"] / f"seed_{cell['seed']}"
    errors: list[str] = []
    required = list(protocol["output_schema"]["required_run_files"])
    required += (
        protocol["output_schema"]["deep_only_files"]
        if cell["model_kind"] == "deep"
        else protocol["output_schema"]["random_forest_only_files"]
    )
    expected_names = set(required)
    if not run_dir.is_dir():
        return (
            {**cell, "run_dir": str(run_dir), "status": "failed", "errors": "missing_run_dir"},
            {},
            {},
        )
    actual_names = {path.name for path in run_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        errors.append(
            "file_set_mismatch:missing="
            + ",".join(sorted(expected_names - actual_names))
            + ":extra="
            + ",".join(sorted(actual_names - expected_names))
        )
    try:
        manifest = read_json(run_dir / "run_manifest.json")
        result = read_json(run_dir / "result.json")
        config = read_json(run_dir / "config_snapshot.json")
        environment = read_json(run_dir / "environment.json")
    except Exception as error:
        return (
            {
                **cell,
                "run_dir": str(run_dir),
                "status": "failed",
                "errors": f"json_error:{type(error).__name__}:{error}",
            },
            {},
            {},
        )
    expected_identity = {
        "protocol_sha256": protocol_sha,
        "model_id": cell["model_id"],
        "fold_id": cell["fold_id"],
        "seed": cell["seed"],
    }
    for key, value in expected_identity.items():
        if manifest.get(key) != value:
            errors.append(f"manifest_identity:{key}")
        if result.get(key) != value:
            errors.append(f"result_identity:{key}")
        if config.get(key) != value:
            errors.append(f"config_identity:{key}")
    if manifest.get("scope") != "formal_v3_run" or manifest.get("completed") is not True:
        errors.append("manifest_not_completed_formal")
    if manifest.get("model_kind") != cell["model_kind"] or result.get("model_kind") != cell["model_kind"]:
        errors.append("model_kind_mismatch")
    if manifest.get("attempt") != 1 or config.get("attempt") != 1:
        errors.append("unexpected_attempt")
    if config.get("protocol") != protocol:
        errors.append("config_protocol_snapshot_mismatch")
    expected_device = "cpu" if cell["model_kind"] == "random_forest" else config.get("device")
    if cell["model_kind"] == "deep" and expected_device not in {"cuda:0", "cuda:1"}:
        errors.append("deep_device_invalid")
    if environment.get("device") != expected_device:
        errors.append("environment_device_mismatch")
    hashes = manifest.get("file_sha256", {})
    expected_hashed = expected_names - {"run_manifest.json"}
    if set(hashes) != expected_hashed:
        errors.append("manifest_hash_file_set_mismatch")
    hashes_ok = True
    for name in sorted(expected_hashed & set(hashes)):
        if sha256(run_dir / name) != hashes[name]:
            hashes_ok = False
            errors.append(f"sha256_mismatch:{name}")

    y = fold_arrays["y"]
    max_metric_delta = 0.0
    probability_checks = True
    recomputed_roles: dict[str, dict] = {}
    for role, filename in (
        ("validation", "validation_probability.npy"),
        ("test", "test_probability.npy"),
    ):
        probability = np.load(run_dir / filename, mmap_mode="r")
        mask = fold_arrays[role]
        if probability.shape != y.shape:
            errors.append(f"{role}_shape_mismatch")
            probability_checks = False
            continue
        if not np.issubdtype(probability.dtype, np.floating):
            errors.append(f"{role}_dtype_not_float")
            probability_checks = False
        if not np.isfinite(probability).all():
            errors.append(f"{role}_nonfinite")
            probability_checks = False
        minimum = float(np.min(probability))
        maximum = float(np.max(probability))
        if minimum < 0.0 or maximum > 1.0:
            errors.append(f"{role}_outside_probability_range")
            probability_checks = False
        if cell["model_kind"] == "deep":
            role_code = 2 if role == "validation" else 3
            prediction_support = fold_arrays["partition"] == role_code
        else:
            prediction_support = mask
        if np.count_nonzero(probability[~prediction_support]) != 0:
            errors.append(f"{role}_nonzero_outside_prediction_support")
            probability_checks = False
        role_metrics = metrics(y, probability, mask)
        recomputed_roles[role] = role_metrics
        delta, role_errors = metric_delta(result.get(role, {}), role_metrics)
        max_metric_delta = max(max_metric_delta, delta)
        errors.extend(f"{role}_{item}" for item in role_errors)

    pr = np.load(run_dir / "pr_curve.npz")
    if set(pr.files) != {"precision", "recall", "thresholds"}:
        errors.append("pr_curve_keys")
    else:
        precision = pr["precision"]
        recall = pr["recall"]
        thresholds = pr["thresholds"]
        if len(precision) != len(recall) or len(thresholds) + 1 != len(precision):
            errors.append("pr_curve_lengths")
        if not all(np.isfinite(array).all() for array in (precision, recall, thresholds)):
            errors.append("pr_curve_nonfinite")

    history_path = run_dir / "history.csv"
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    diagnostic: dict = {
        "model_id": cell["model_id"],
        "model_kind": cell["model_kind"],
        "fold_id": cell["fold_id"],
        "seed": cell["seed"],
    }
    if cell["model_kind"] == "deep":
        epochs = int(result.get("epochs_completed", -1))
        best_epoch = int(result.get("best_epoch", -1))
        if len(history) != epochs or not (1 <= best_epoch <= epochs <= protocol["deep_training"]["maximum_epochs"]):
            errors.append("deep_history_epoch_mismatch")
        if epochs < protocol["deep_training"]["minimum_epochs_before_early_stop"]:
            errors.append("deep_below_minimum_epochs")
        if int(result.get("trainable_parameters", -1)) != cell["expected_parameters"]:
            errors.append("trainable_parameter_mismatch")
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        for key, value in expected_identity.items():
            if checkpoint.get(key) != value:
                errors.append(f"checkpoint_identity:{key}")
        if int(checkpoint.get("epoch", -1)) != best_epoch:
            errors.append("checkpoint_best_epoch_mismatch")
        if not checkpoint.get("model_state") or not checkpoint.get("optimizer_state"):
            errors.append("checkpoint_state_missing")
        diagnostic.update(
            {
                "epochs_completed": epochs,
                "best_epoch": best_epoch,
                "training_seconds": sum(float(row["seconds"]) for row in history),
                "final_training_loss": float(history[-1]["training_loss"]),
                "best_validation_ap": float(result["validation"]["average_precision"]),
                "training_pixels": "",
                "features": "",
                "trees": "",
            }
        )
    else:
        if len(history) != 1:
            errors.append("rf_history_row_count")
        if int(float(history[0].get("trees", -1))) != protocol["random_forest_training"]["n_estimators"]:
            errors.append("rf_tree_count")
        expected_features = 32 if cell["model_id"] == "rf_direct" else 56
        if int(result.get("features", -1)) != expected_features:
            errors.append("rf_feature_count")
        diagnostic.update(
            {
                "epochs_completed": "",
                "best_epoch": "",
                "training_seconds": float(history[0]["fit_seconds"]),
                "final_training_loss": "",
                "best_validation_ap": float(result["validation"]["average_precision"]),
                "training_pixels": int(result["training_pixels"]),
                "features": int(result["features"]),
                "trees": int(float(history[0]["trees"])),
            }
        )

    individual = {
        "ordinal": cell["ordinal"],
        "model_id": cell["model_id"],
        "model_kind": cell["model_kind"],
        "analysis_tier": cell["analysis_tier"],
        "fold_id": cell["fold_id"],
        "seed": cell["seed"],
    }
    for role in ("validation", "test"):
        for key in METRIC_KEYS:
            individual[f"{role}_{key}"] = recomputed_roles.get(role, {}).get(key, "")
    individual.update(
        {
            "epochs_completed": result.get("epochs_completed", ""),
            "best_epoch": result.get("best_epoch", ""),
            "validation_selected_threshold_descriptive": result.get(
                "validation_selected_threshold_descriptive", ""
            ),
            "validation_selected_f1_descriptive": result.get(
                "validation_selected_f1_descriptive", ""
            ),
        }
    )
    inventory = {
        "ordinal": cell["ordinal"],
        "cell_id": f"{cell['model_id']}__{cell['fold_id']}__seed_{cell['seed']}",
        "model_id": cell["model_id"],
        "model_kind": cell["model_kind"],
        "fold_id": cell["fold_id"],
        "seed": cell["seed"],
        "attempt": manifest.get("attempt", ""),
        "file_count": len(actual_names),
        "bytes": sum(path.stat().st_size for path in run_dir.iterdir() if path.is_file()),
        "manifest_sha256": sha256(run_dir / "run_manifest.json"),
        "hashes_verified": hashes_ok and set(hashes) == expected_hashed,
        "probabilities_verified": probability_checks,
        "maximum_metric_delta": max_metric_delta,
        "status": "passed" if not errors else "failed",
        "errors": " | ".join(errors),
        "run_dir": str(run_dir.relative_to(root)).replace("\\", "/"),
    }
    return inventory, individual, diagnostic


def aggregate_group(
    result_root: Path,
    model_id: str,
    fold_id: str,
    seeds: list[int],
    fold_arrays: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    row = {"model_id": model_id, "fold_id": fold_id, "seed_count": len(seeds)}
    for role, filename in (
        ("validation", "validation_probability.npy"),
        ("test", "test_probability.npy"),
    ):
        accumulator = np.zeros(fold_arrays["y"].shape, dtype=np.float64)
        for seed in seeds:
            accumulator += np.load(
                result_root / model_id / fold_id / f"seed_{seed}" / filename,
                mmap_mode="r",
            )
        ensemble = accumulator / len(seeds)
        arrays[role] = ensemble
        role_metrics = metrics(fold_arrays["y"], ensemble, fold_arrays[role])
        for key, value in role_metrics.items():
            row[f"{role}_{key}"] = value
    return row, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol_path = root / PROTOCOL_REL
    lock_path = root / LOCK_REL
    protocol = read_json(protocol_path)
    protocol_sha = sha256(protocol_path)
    lock = read_json(lock_path)
    if lock.get("protocol_sha256") != protocol_sha or lock.get("status") != "locked":
        raise RuntimeError("Protocol lock mismatch")
    result_root = root / "01_data/interim/results_v3" / protocol_sha
    log_root = root / EXECUTION_SUMMARY_REL / protocol_sha
    execution_summary_path = log_root / "scheduler_execution_summary.json"
    if not result_root.is_dir() or not execution_summary_path.is_file():
        raise FileNotFoundError("Formal results or scheduler execution summary missing")
    execution_summary = read_json(execution_summary_path)
    if execution_summary.get("completed") != 128 or execution_summary.get("failed") != 0:
        raise RuntimeError("Scheduler execution summary is not 128 completed / 0 failed")
    audit_dir = root / AUDIT_REL
    if audit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {audit_dir}")
    audit_dir.mkdir(parents=True)

    cells = expected_cells(protocol)
    fold_cache = {
        fold_id: load_fold_arrays(root, fold_id) for fold_id in protocol["data"]["folds"]
    }
    inventory_rows: list[dict] = []
    individual_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for index, cell in enumerate(cells, 1):
        inventory, individual, diagnostic = validate_run(
            root,
            result_root,
            protocol,
            protocol_sha,
            cell,
            fold_cache[cell["fold_id"]],
        )
        inventory_rows.append(inventory)
        if individual:
            individual_rows.append(individual)
        if diagnostic:
            diagnostic_rows.append(diagnostic)
        if index % 8 == 0:
            print(f"validated {index}/128", flush=True)

    failed_runs = [row for row in inventory_rows if row["status"] != "passed"]
    write_csv(audit_dir / "run_inventory_v3.csv", inventory_rows)
    if failed_runs:
        write_json(
            audit_dir / "stage9_acceptance_v3.json",
            {
                "overall_accepted": False,
                "failed_runs": failed_runs,
                "validated_runs": len(inventory_rows),
                "protocol_sha256": protocol_sha,
            },
        )
        raise RuntimeError(f"Run validation failed for {len(failed_runs)} cells")

    write_csv(audit_dir / "individual_metrics_v3.csv", individual_rows)
    write_csv(audit_dir / "training_diagnostics_v3.csv", diagnostic_rows)
    ensemble_rows: list[dict] = []
    ensemble_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    model_tiers = {
        item["model_id"]: item["analysis_tier"]
        for group in (protocol["deep_models"], protocol["random_forest_models"])
        for item in group
    }
    model_order = [item["model_id"] for item in protocol["deep_models"]] + [
        item["model_id"] for item in protocol["random_forest_models"]
    ]
    for model_id in model_order:
        for fold_id in protocol["data"]["folds"]:
            row, arrays = aggregate_group(
                result_root,
                model_id,
                fold_id,
                [int(seed) for seed in protocol["seeds"]],
                fold_cache[fold_id],
            )
            row = {
                "model_id": model_id,
                "analysis_tier": model_tiers[model_id],
                **{key: value for key, value in row.items() if key != "model_id"},
            }
            ensemble_rows.append(row)
            ensemble_arrays[(model_id, fold_id)] = arrays
    write_csv(audit_dir / "fold_seed_ensemble_metrics_v3.csv", ensemble_rows)

    pooled_rows: list[dict] = []
    model_summary_rows: list[dict] = []
    for model_id in model_order:
        fold_rows = [row for row in ensemble_rows if row["model_id"] == model_id]
        pooled_truth: list[np.ndarray] = []
        pooled_score: list[np.ndarray] = []
        for fold_id in protocol["data"]["folds"]:
            data = fold_cache[fold_id]
            selected = data["test"] & (data["y"] != 2)
            pooled_truth.append((data["y"][selected] == 1).astype(np.int8))
            pooled_score.append(ensemble_arrays[(model_id, fold_id)]["test"][selected])
        truth = np.concatenate(pooled_truth)
        score = np.concatenate(pooled_score)
        pooled_y = truth
        pooled_mask = np.ones(truth.shape, dtype=bool)
        pooled_metric = metrics(pooled_y, score, pooled_mask)
        pooled_rows.append(
            {
                "model_id": model_id,
                "analysis_tier": model_tiers[model_id],
                **pooled_metric,
            }
        )
        summary = {
            "model_id": model_id,
            "analysis_tier": model_tiers[model_id],
            "fold_count": 4,
            "seed_count_per_fold": 4,
        }
        for metric_name in SUMMARY_METRICS:
            values = np.asarray(
                [float(row[f"test_{metric_name}"]) for row in fold_rows], dtype=np.float64
            )
            summary[f"fold_ensemble_{metric_name}_mean"] = float(values.mean())
            summary[f"fold_ensemble_{metric_name}_sd"] = float(values.std(ddof=1))
        individual_ap = np.asarray(
            [
                float(row["test_average_precision"])
                for row in individual_rows
                if row["model_id"] == model_id
            ],
            dtype=np.float64,
        )
        summary["individual_test_ap_mean"] = float(individual_ap.mean())
        summary["individual_test_ap_sd"] = float(individual_ap.std(ddof=1))
        summary["pooled_oof_average_precision"] = pooled_metric["average_precision"]
        summary["pooled_oof_roc_auc"] = pooled_metric["roc_auc"]
        model_summary_rows.append(summary)

    by_ap = sorted(
        model_summary_rows,
        key=lambda row: row["fold_ensemble_average_precision_mean"],
        reverse=True,
    )
    overall_rank = {row["model_id"]: rank for rank, row in enumerate(by_ap, 1)}
    primary = [row for row in by_ap if row["analysis_tier"] == "primary_capacity_matched"]
    primary_rank = {row["model_id"]: rank for rank, row in enumerate(primary, 1)}
    for row in model_summary_rows:
        row["overall_descriptive_rank"] = overall_rank[row["model_id"]]
        row["primary_capacity_matched_rank"] = primary_rank.get(row["model_id"], "")
    model_summary_rows.sort(key=lambda row: overall_rank[row["model_id"]])
    write_csv(audit_dir / "model_summary_v3.csv", model_summary_rows)
    write_csv(audit_dir / "pooled_oof_metrics_v3.csv", pooled_rows)

    comparisons = [
        ("temporal_attn_vs_unet", "temporal_attn", "unet_matched", "primary_architecture"),
        ("temporal_attn_vs_concat", "temporal_attn", "temporal_concat", "primary_architecture"),
        ("temporal_attn_vs_summer", "temporal_attn", "summer_only", "input_ablation"),
        ("temporal_attn_vs_thermal", "temporal_attn", "thermal_only", "input_ablation"),
        ("temporal_attn_vs_optical", "temporal_attn", "optical_only", "input_ablation"),
        ("rf_context_vs_direct", "rf_context", "rf_direct", "context_sensitivity"),
    ]
    ensemble_lookup = {
        (row["model_id"], row["fold_id"]): row for row in ensemble_rows
    }
    difference_rows: list[dict] = []
    comparison_summary_rows: list[dict] = []
    for comparison_id, left, right, boundary in comparisons:
        differences = []
        for fold_id in protocol["data"]["folds"]:
            left_ap = float(ensemble_lookup[(left, fold_id)]["test_average_precision"])
            right_ap = float(ensemble_lookup[(right, fold_id)]["test_average_precision"])
            difference = left_ap - right_ap
            differences.append(difference)
            difference_rows.append(
                {
                    "comparison_id": comparison_id,
                    "interpretation_boundary": boundary,
                    "fold_id": fold_id,
                    "left_model": left,
                    "right_model": right,
                    "left_test_ap": left_ap,
                    "right_test_ap": right_ap,
                    "ap_difference_left_minus_right": difference,
                }
            )
        difference_array = np.asarray(differences, dtype=np.float64)
        comparison_summary_rows.append(
            {
                "comparison_id": comparison_id,
                "interpretation_boundary": boundary,
                "left_model": left,
                "right_model": right,
                "folds": 4,
                "mean_ap_difference_left_minus_right": float(difference_array.mean()),
                "sd_ap_difference": float(difference_array.std(ddof=1)),
                "left_wins": int((difference_array > 0).sum()),
                "ties": int((difference_array == 0).sum()),
                "right_wins": int((difference_array < 0).sum()),
            }
        )
    write_csv(audit_dir / "pairwise_fold_ap_differences_v3.csv", difference_rows)
    write_csv(audit_dir / "pairwise_summary_v3.csv", comparison_summary_rows)

    combined_manifest_records = [
        f"{row['cell_id']} {row['manifest_sha256']}" for row in inventory_rows
    ]
    input_digest = sha256_text("\n".join(combined_manifest_records) + "\n")
    aggregation_sources = {
        "aggregation_script": root
        / "02_scripts/analysis/23_validate_and_aggregate_results_v3.py",
        "formal_runner": root / "02_scripts/modeling/13_run_experiment_v3.py",
        "scheduler": root / "02_scripts/modeling/14_schedule_experiments_v3.py",
        "dataset_loader": root / "02_scripts/modeling/dataset_common_v3.py",
        "protocol": protocol_path,
        "scheduler_execution_summary": execution_summary_path,
    }
    fold_input_hashes = {
        f"{fold_id}/{name}": sha256(
            root / "01_data/interim/dataset_v3" / fold_id / name
        )
        for fold_id in protocol["data"]["folds"]
        for name in ("y.npy", "partition.npy", "val.npy", "test.npy")
    }
    write_json(
        audit_dir / "input_manifest_digest_v3.json",
        {
            "protocol_sha256": protocol_sha,
            "run_manifests": 128,
            "ordered_manifest_digest_sha256": input_digest,
            "scheduler_execution_summary_sha256": sha256(execution_summary_path),
            "total_input_bytes": int(sum(row["bytes"] for row in inventory_rows)),
            "aggregation_source_sha256": {
                name: sha256(path) for name, path in aggregation_sources.items()
            },
            "fold_input_sha256": fold_input_hashes,
        },
    )
    acceptance = {
        "overall_accepted": True,
        "status": "formal_runs_validated_and_aggregated",
        "protocol_sha256": protocol_sha,
        "validated_runs": 128,
        "passed_runs": 128,
        "failed_runs": 0,
        "run_manifests": 128,
        "result_json_files": 128,
        "remaining_attempt_directories": 0,
        "scheduler_completed": execution_summary["completed"],
        "scheduler_failed": execution_summary["failed"],
        "fold_seed_ensembles": 32,
        "model_summaries": 8,
        "primary_comparison_boundary": "Only the three primary_capacity_matched models support direct architecture attribution.",
        "claim_boundary": "Metrics rank proximity to QFFD traces coded Inferred under a proxy label; they do not establish concealed-fault discovery.",
        "next_gate": "Independently recompute aggregates before interpretation or release.",
    }
    write_json(audit_dir / "stage9_acceptance_v3.json", acceptance)
    (audit_dir / "README.md").write_text(
        "# V3 formal-run validation and aggregation — stage 9\n\n"
        "Status: 128/128 formal cells validated and aggregated.\n\n"
        f"Protocol SHA-256: `{protocol_sha}`\n\n"
        "Primary summaries use the arithmetic mean of four seed probability maps "
        "within each model-fold cell, followed by an equal-weight mean and sample "
        "standard deviation across the four spatial folds. Pooled OOF metrics are "
        "supplementary. No pixelwise bootstrap interval is reported.\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_names = sorted(
        path.name
        for path in audit_dir.iterdir()
        if path.is_file() and path.name != "aggregate_checksums_v3.sha256"
    )
    with (audit_dir / "aggregate_checksums_v3.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for name in checksum_names:
            handle.write(f"{sha256(audit_dir / name)}  {name}\n")
    print(
        json.dumps(
            {
                "overall_accepted": True,
                "validated_runs": 128,
                "fold_seed_ensembles": 32,
                "model_summaries": 8,
                "ordered_manifest_digest_sha256": input_digest,
                "audit_dir": str(audit_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
