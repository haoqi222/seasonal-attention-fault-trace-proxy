#!/usr/bin/env python3
"""Validate V4.2 re-inference outputs and recompute the complete 128-cell comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


V4_PROTOCOL = Path("03_configs/inference_v4_2_protocol.json")
V3_PROTOCOL = Path("03_configs/experiment_v3_protocol.json")
OUTPUT = Path("04_results/experiment_v4_aggregate")
SUMMARY_METRICS = [
    "average_precision", "roc_auc", "precision_top_1pct", "recall_top_1pct",
    "precision_top_5pct", "recall_top_5pct", "f1_at_0_5",
    "precision_at_0_5", "recall_at_0_5",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(np.asarray(probability[selected], dtype=np.float64), 0, 1)
    result = {
        "n": int(len(truth)), "positive": int(truth.sum()),
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
    result.update({
        "threshold": 0.5,
        "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
        "precision_at_0_5": precision, "recall_at_0_5": recall,
        "tp_at_0_5": tp, "fp_at_0_5": fp, "fn_at_0_5": fn,
    })
    return result


def max_metric_delta(stored: dict, recalculated: dict) -> float:
    integer = {"n", "positive", "tp_at_0_5", "fp_at_0_5", "fn_at_0_5"}
    maximum = 0.0
    for name, value in recalculated.items():
        if name in integer:
            if int(stored[name]) != int(value):
                raise ValueError(f"Integer metric mismatch {name}: {stored[name]} != {value}")
        else:
            maximum = max(maximum, abs(float(stored[name]) - float(value)))
    if maximum > 5e-6:
        raise ValueError(f"Metric delta exceeds tolerance: {maximum}")
    return maximum


def fold_arrays(root: Path, fold: str) -> dict:
    folder = root / "01_data/interim/dataset_v3" / fold
    return {
        "y": np.load(folder / "y.npy", mmap_mode="r"),
        "partition": np.load(folder / "partition.npy", mmap_mode="r"),
        "validation": np.load(folder / "val.npy", mmap_mode="r").astype(bool),
        "test": np.load(folder / "test.npy", mmap_mode="r").astype(bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    v4_path = root / V4_PROTOCOL
    v4 = read_json(v4_path)
    v4_sha = sha256(v4_path)
    v3 = read_json(root / V3_PROTOCOL)
    v3_sha = v4["parent_v3"]["training_protocol_sha256"]
    deep_root = root / "01_data/interim/results_v4" / v4_sha
    rf_root = root / "01_data/interim/results_v3" / v3_sha
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    preflight = read_json(root / "04_results/inference_v4_preflight" / v4_sha / "stage11_preflight_acceptance_v4.json")
    if preflight["status"] != "accepted" or preflight["test_probability_files"] != 0:
        raise RuntimeError("V4.2 validation-only preflight is not accepted")

    folds = list(v3["data"]["folds"])
    seeds = [int(value) for value in v3["seeds"]]
    deep_models = [item["model_id"] for item in v3["deep_models"]]
    rf_models = [item["model_id"] for item in v3["random_forest_models"]]
    order = deep_models + rf_models
    tiers = {item["model_id"]: item["analysis_tier"] for group in (v3["deep_models"], v3["random_forest_models"]) for item in group}
    cache = {fold: fold_arrays(root, fold) for fold in folds}
    inventory = []
    individual = []
    maximum_delta = 0.0

    for model in order:
        for fold in folds:
            data = cache[fold]
            for seed in seeds:
                deep = model in deep_models
                cell = (deep_root if deep else rf_root) / model / fold / f"seed_{seed}"
                errors = []
                expected = ({"validation_probability.npy", "test_probability.npy", "pr_curve.npz", "result.json",
                             "environment.json", "config_snapshot.json", "run_manifest.json"} if deep else
                            set(v3["output_schema"]["required_run_files"] + v3["output_schema"]["random_forest_only_files"]))
                actual = {path.name for path in cell.iterdir() if path.is_file()} if cell.is_dir() else set()
                if actual != expected:
                    errors.append("file_set")
                try:
                    manifest = read_json(cell / "run_manifest.json")
                    result = read_json(cell / "result.json")
                    if deep:
                        if manifest["v4_protocol_sha256"] != v4_sha or manifest["preflight"] or manifest["roles"] != ["validation", "test"]:
                            errors.append("v4_manifest_identity")
                        for name, digest in manifest["files"].items():
                            if sha256(cell / name) != digest:
                                errors.append("hash:" + name)
                        if read_json(cell / "config_snapshot.json") != v4:
                            errors.append("config_snapshot")
                        checkpoint = root / result["parent_checkpoint_path"]
                        if sha256(checkpoint) != result["parent_checkpoint_sha256"]:
                            errors.append("parent_checkpoint_hash")
                    else:
                        if not manifest["completed"] or manifest["protocol_sha256"] != v3_sha:
                            errors.append("v3_rf_manifest_identity")
                        for name, digest in manifest["file_sha256"].items():
                            if sha256(cell / name) != digest:
                                errors.append("hash:" + name)
                    row = {"model_id": model, "model_kind": "deep" if deep else "random_forest",
                           "analysis_tier": tiers[model], "fold_id": fold, "seed": seed,
                           "source_protocol": "v4.2_weighted" if deep else "v3_exact_reuse"}
                    for role, role_code in (("validation", 2), ("test", 3)):
                        probability = np.load(cell / f"{role}_probability.npy", mmap_mode="r")
                        if probability.shape != data["y"].shape or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                            errors.append(role + "_array")
                        support = data["partition"] == role_code if deep else data[role]
                        if np.count_nonzero(probability[~support]) != 0:
                            errors.append(role + "_support")
                        calculated = metrics(data["y"], probability, data[role])
                        maximum_delta = max(maximum_delta, max_metric_delta(result[role], calculated))
                        for key, value in calculated.items():
                            row[f"{role}_{key}"] = value
                    if deep:
                        parent_result = read_json(rf_root / model / fold / f"seed_{seed}" / "result.json")
                        row["epochs_completed"] = parent_result["epochs_completed"]
                        row["best_epoch"] = parent_result["best_epoch"]
                    else:
                        row["epochs_completed"] = ""
                        row["best_epoch"] = ""
                    individual.append(row)
                except Exception as exc:
                    errors.append(type(exc).__name__ + ":" + str(exc))
                inventory.append({
                    "model_id": model, "fold_id": fold, "seed": seed,
                    "source_protocol": "v4.2_weighted" if deep else "v3_exact_reuse",
                    "run_dir": str(cell.relative_to(root)).replace("\\", "/"),
                    "bytes": sum(path.stat().st_size for path in cell.iterdir() if path.is_file()) if cell.is_dir() else 0,
                    "status": "passed" if not errors else "failed", "errors": " | ".join(errors),
                })
    write_csv(output / "run_inventory_v4.csv", inventory)
    failures = [row for row in inventory if row["status"] != "passed"]
    if failures:
        write_json(output / "stage12_acceptance_v4.json", {"overall_accepted": False, "failures": failures})
        raise RuntimeError(f"V4 validation failed: {len(failures)} cells")
    write_csv(output / "individual_metrics_v4.csv", individual)

    def cell_path(model: str, fold: str, seed: int) -> Path:
        return (deep_root if model in deep_models else rf_root) / model / fold / f"seed_{seed}"

    ensemble_rows = []
    ensemble_arrays = {}
    for model in order:
        for fold in folds:
            data = cache[fold]
            record = {"model_id": model, "analysis_tier": tiers[model], "fold_id": fold, "seed_count": 4}
            arrays = {}
            for role in ("validation", "test"):
                total = np.zeros(data["y"].shape, dtype=np.float64)
                for seed in seeds:
                    total += np.load(cell_path(model, fold, seed) / f"{role}_probability.npy", mmap_mode="r")
                arrays[role] = total / 4
                for key, value in metrics(data["y"], arrays[role], data[role]).items():
                    record[f"{role}_{key}"] = value
            ensemble_rows.append(record)
            ensemble_arrays[(model, fold)] = arrays
    write_csv(output / "fold_seed_ensemble_metrics_v4.csv", ensemble_rows)

    summaries = []
    pooled = []
    for model in order:
        records = [row for row in ensemble_rows if row["model_id"] == model]
        truth_parts, score_parts = [], []
        for fold in folds:
            data = cache[fold]
            selected = data["test"] & (data["y"] != 2)
            truth_parts.append((data["y"][selected] == 1).astype(np.int8))
            score_parts.append(ensemble_arrays[(model, fold)]["test"][selected])
        truth, score = np.concatenate(truth_parts), np.concatenate(score_parts)
        pooled_metric = metrics(truth, score, np.ones(truth.shape, dtype=bool))
        pooled.append({"model_id": model, "analysis_tier": tiers[model], **pooled_metric})
        summary = {"model_id": model, "analysis_tier": tiers[model], "fold_count": 4, "seed_count_per_fold": 4}
        for name in SUMMARY_METRICS:
            values = np.array([float(row[f"test_{name}"]) for row in records])
            summary[f"fold_ensemble_{name}_mean"] = float(values.mean())
            summary[f"fold_ensemble_{name}_sd"] = float(values.std(ddof=1))
        aps = np.array([float(row["test_average_precision"]) for row in individual if row["model_id"] == model])
        summary["individual_test_ap_mean"] = float(aps.mean())
        summary["individual_test_ap_sd"] = float(aps.std(ddof=1))
        summary["pooled_oof_average_precision"] = pooled_metric["average_precision"]
        summary["pooled_oof_roc_auc"] = pooled_metric["roc_auc"]
        summaries.append(summary)
    ranked = sorted(summaries, key=lambda row: row["fold_ensemble_average_precision_mean"], reverse=True)
    overall_rank = {row["model_id"]: index for index, row in enumerate(ranked, 1)}
    primary_ranked = [row for row in ranked if row["analysis_tier"] == "primary_capacity_matched"]
    primary_rank = {row["model_id"]: index for index, row in enumerate(primary_ranked, 1)}
    for row in summaries:
        row["overall_descriptive_rank"] = overall_rank[row["model_id"]]
        row["primary_capacity_matched_rank"] = primary_rank.get(row["model_id"], "")
    summaries.sort(key=lambda row: overall_rank[row["model_id"]])
    write_csv(output / "model_summary_v4.csv", summaries)
    write_csv(output / "pooled_oof_metrics_v4.csv", pooled)

    comparisons = [
        ("temporal_attn_vs_unet", "temporal_attn", "unet_matched", "primary_architecture"),
        ("temporal_attn_vs_concat", "temporal_attn", "temporal_concat", "primary_architecture"),
        ("temporal_attn_vs_summer", "temporal_attn", "summer_only", "input_ablation"),
        ("temporal_attn_vs_thermal", "temporal_attn", "thermal_only", "input_ablation"),
        ("temporal_attn_vs_optical", "temporal_attn", "optical_only", "input_ablation"),
        ("rf_context_vs_direct", "rf_context", "rf_direct", "context_sensitivity"),
    ]
    lookup = {(row["model_id"], row["fold_id"]): row for row in ensemble_rows}
    differences, pairwise = [], []
    for comparison_id, left, right, boundary in comparisons:
        values = []
        for fold in folds:
            left_ap = float(lookup[(left, fold)]["test_average_precision"])
            right_ap = float(lookup[(right, fold)]["test_average_precision"])
            difference = left_ap - right_ap
            values.append(difference)
            differences.append({"comparison_id": comparison_id, "interpretation_boundary": boundary,
                                "fold_id": fold, "left_model": left, "right_model": right,
                                "left_test_ap": left_ap, "right_test_ap": right_ap,
                                "ap_difference_left_minus_right": difference})
        values = np.asarray(values)
        pairwise.append({"comparison_id": comparison_id, "interpretation_boundary": boundary,
                         "left_model": left, "right_model": right, "folds": 4,
                         "mean_ap_difference_left_minus_right": float(values.mean()),
                         "sd_ap_difference": float(values.std(ddof=1)),
                         "left_wins": int((values > 0).sum()), "ties": int((values == 0).sum()),
                         "right_wins": int((values < 0).sum())})
    write_csv(output / "pairwise_fold_ap_differences_v4.csv", differences)
    write_csv(output / "pairwise_summary_v4.csv", pairwise)

    v3_summary = {row["model_id"]: row for row in read_csv(root / "04_results/experiment_v3_aggregate/model_summary_v3.csv")}
    v3_fold = {(row["model_id"], row["fold_id"]): row for row in read_csv(root / "04_results/experiment_v3_aggregate/fold_seed_ensemble_metrics_v3.csv")}
    change_rows = []
    for model in order:
        current = next(row for row in summaries if row["model_id"] == model)
        change_rows.append({
            "model_id": model, "source_protocol": "v4.2_weighted" if model in deep_models else "v3_exact_reuse",
            "v3_mean_ap": float(v3_summary[model]["fold_ensemble_average_precision_mean"]),
            "v4_mean_ap": float(current["fold_ensemble_average_precision_mean"]),
            "v4_minus_v3_mean_ap": float(current["fold_ensemble_average_precision_mean"]) - float(v3_summary[model]["fold_ensemble_average_precision_mean"]),
            "v3_rank": int(v3_summary[model]["overall_descriptive_rank"]), "v4_rank": int(current["overall_descriptive_rank"]),
        })
    write_csv(output / "v3_v4_model_changes.csv", change_rows)
    fold_changes = []
    for row in ensemble_rows:
        previous = v3_fold[(row["model_id"], row["fold_id"])]
        fold_changes.append({"model_id": row["model_id"], "fold_id": row["fold_id"],
                             "v3_test_ap": float(previous["test_average_precision"]),
                             "v4_test_ap": float(row["test_average_precision"]),
                             "v4_minus_v3_test_ap": float(row["test_average_precision"]) - float(previous["test_average_precision"])})
    write_csv(output / "v3_v4_fold_ap_changes.csv", fold_changes)

    acceptance = {
        "overall_accepted": True, "status": "v4_2_weighted_inference_validated_and_aggregated",
        "v4_protocol_sha256": v4_sha, "parent_v3_protocol_sha256": v3_sha,
        "deep_v4_cells": 96, "rf_v3_exact_reuse_cells": 32, "total_comparison_cells": 128,
        "fold_seed_ensembles": 32, "model_summaries": 8,
        "maximum_individual_metric_recalculation_delta": maximum_delta,
        "claim_boundary": v4["scientific_boundaries"],
        "next_gate": "Independently verify all aggregates and quantify V4 test-map seam ratios before release use.",
    }
    write_json(output / "stage12_acceptance_v4.json", acceptance)
    (output / "README.md").write_text(
        "# V4.2 weighted-inference aggregation\n\n"
        "Six deep models use V4.2 center-weighted re-inference from frozen V3 checkpoints. "
        "The two random-forest models reuse their exact V3 probability arrays because they do not use sliding windows.\n",
        encoding="utf-8", newline="\n")
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name != "aggregate_checksums_v4.sha256")
    (output / "aggregate_checksums_v4.sha256").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="ascii", newline="\n")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
