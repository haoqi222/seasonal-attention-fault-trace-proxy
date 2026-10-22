#!/usr/bin/env python3
"""Validate and aggregate the 48-cell random-holdout diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PROTOCOL = Path("03_configs/random_holdout_v1_protocol.json")
OUTPUT = Path("04_results/random_holdout_v1_aggregate")
METRICS = ["average_precision", "roc_auc", "precision_top_1pct", "recall_top_1pct",
           "precision_top_5pct", "recall_top_5pct", "f1_at_0_5",
           "precision_at_0_5", "recall_at_0_5"]


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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    truth = (y[mask] == 1).astype(np.int8)
    score = np.clip(np.asarray(probability[mask], dtype=np.float64), 0, 1)
    result = {"n": int(len(truth)), "positive": int(truth.sum()),
              "positive_rate": float(truth.mean()),
              "average_precision": float(average_precision_score(truth, score)),
              "roc_auc": float(roc_auc_score(truth, score))}
    for fraction in (0.01, 0.05):
        count = max(1, int(math.ceil(len(score) * fraction)))
        top = np.argpartition(score, -count)[-count:]
        tp = int(truth[top].sum())
        label = f"top_{int(fraction * 100)}pct"
        result[f"precision_{label}"] = tp / count
        result[f"recall_{label}"] = tp / max(int(truth.sum()), 1)
    predicted = score >= 0.5
    tp = int((predicted & (truth == 1)).sum())
    fp = int((predicted & (truth == 0)).sum())
    fn = int((~predicted & (truth == 1)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result.update({"threshold": 0.5,
                   "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
                   "precision_at_0_5": precision, "recall_at_0_5": recall,
                   "tp_at_0_5": tp, "fp_at_0_5": fp, "fn_at_0_5": fn})
    return result


def metric_delta(stored: dict, calculated: dict) -> float:
    integer = {"n", "positive", "tp_at_0_5", "fp_at_0_5", "fn_at_0_5"}
    maximum = 0.0
    for name, value in calculated.items():
        if name in integer:
            if int(stored[name]) != int(value):
                raise ValueError(f"Integer metric mismatch: {name}")
        else:
            maximum = max(maximum, abs(float(stored[name]) - float(value)))
    if maximum > 5e-12:
        raise ValueError(f"Metric mismatch exceeds tolerance: {maximum}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol_path = root / PROTOCOL
    protocol, protocol_sha = read_json(protocol_path), sha256(protocol_path)
    result_root = root / "01_data/interim/results_random_holdout_v1" / protocol_sha
    prepared_root = root / "01_data/interim/random_holdout_v1" / protocol_sha
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    build_acceptance = read_json(prepared_root / "build_acceptance.json")
    if build_acceptance.get("status") != "accepted" or build_acceptance.get("split_count") != 4:
        raise RuntimeError("Random split build is not accepted")
    v4_acceptance = read_json(root / "04_results/experiment_v4_aggregate/stage12_acceptance_v4.json")
    v4_verification = read_json(root / "04_results/experiment_v4_aggregate/independent_verification_v4.json")
    if not v4_acceptance.get("overall_accepted"):
        raise RuntimeError("V4.2 spatial aggregate is unavailable")
    known_exception = (v4_verification.get("checks_total") == 793 and
                       v4_verification.get("checks_passed") == 792 and
                       len(v4_verification.get("failures", [])) == 1 and
                       v4_verification["failures"][0].get("name") == "formal_test_severe_cases_halved")
    if not known_exception:
        raise RuntimeError("V4.2 verification state differs from the disclosed 792/793 boundary")

    split_records = {item["split_id"]: item for item in protocol["splits"]}
    models, seeds = list(protocol["models"]), [int(x) for x in protocol["seeds"]]
    data = {}
    for split_id, record in split_records.items():
        source = root / "01_data/interim/dataset_v3" / record["source_fold"]
        prepared = prepared_root / split_id
        manifest = read_json(prepared / "split_manifest.json")
        if manifest.get("status") != "accepted" or manifest.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"Unaccepted prepared split: {split_id}")
        paths = {"y": source / "y.npy", "train": prepared / "train.npy",
                 "validation": prepared / "val.npy", "test": prepared / "test.npy",
                 "normalization": prepared / "norm_stats.json"}
        for name, path in paths.items():
            if sha256(path) != manifest["file_sha256"][name]:
                raise RuntimeError(f"Prepared data hash mismatch: {split_id} {name}")
        data[split_id] = {"y": np.load(paths["y"], mmap_mode="r"),
                          "validation": np.load(paths["validation"], mmap_mode="r").astype(bool),
                          "test": np.load(paths["test"], mmap_mode="r").astype(bool)}

    expected_files = {"checkpoint.pt", "config_snapshot.json", "environment.json", "history.csv",
                      "probability.npy", "pr_curve.npz", "result.json", "run_manifest.json"}
    inventory, individual, max_delta = [], [], 0.0
    for model in models:
        for split_id in split_records:
            arrays = data[split_id]
            for seed in seeds:
                cell = result_root / model / split_id / f"seed_{seed}"
                errors = []
                actual = {p.name for p in cell.iterdir() if p.is_file()} if cell.is_dir() else set()
                if actual != expected_files:
                    errors.append("file_set")
                try:
                    manifest = read_json(cell / "run_manifest.json")
                    result = read_json(cell / "result.json")
                    identity = {"status": "complete", "scope": "formal_random_holdout_diagnostic_v1",
                                "protocol_sha256": protocol_sha, "model_id": model,
                                "split_id": split_id, "seed": seed}
                    for key, value in identity.items():
                        if manifest.get(key) != value:
                            errors.append("manifest_" + key)
                    for name, digest in manifest.get("file_sha256", {}).items():
                        if sha256(cell / name) != digest:
                            errors.append("hash:" + name)
                    if set(manifest.get("file_sha256", {})) != expected_files - {"run_manifest.json"}:
                        errors.append("manifest_file_list")
                    snapshot = read_json(cell / "config_snapshot.json")
                    if snapshot.get("protocol") != protocol or snapshot.get("protocol_sha256") != protocol_sha:
                        errors.append("config_snapshot")
                    if any(result.get(key) != value for key, value in identity.items() if key != "status"):
                        errors.append("result_identity")
                    if not result.get("known_spatial_context_overlap") or not result.get("test_labels_accessed"):
                        errors.append("diagnostic_boundary")
                    checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=False)
                    if any(checkpoint.get(key) != value for key, value in {
                        "protocol_sha256": protocol_sha, "model_id": model,
                        "split_id": split_id, "seed": seed}.items()):
                        errors.append("checkpoint_identity")
                    probability = np.load(cell / "probability.npy", mmap_mode="r")
                    if probability.shape != arrays["y"].shape or not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                        errors.append("probability_array")
                    row = {"model_id": model, "split_id": split_id,
                           "source_fold": split_records[split_id]["source_fold"], "seed": seed,
                           "epochs_completed": result["epochs_completed"], "best_epoch": result["best_epoch"],
                           "elapsed_seconds": result["elapsed_seconds"]}
                    for role in ("validation", "test"):
                        calculated = metrics(arrays["y"], probability, arrays[role])
                        max_delta = max(max_delta, metric_delta(result[role], calculated))
                        for key, value in calculated.items():
                            row[f"{role}_{key}"] = value
                    individual.append(row)
                except Exception as error:
                    errors.append(type(error).__name__ + ":" + str(error))
                inventory.append({"model_id": model, "split_id": split_id, "seed": seed,
                                  "run_dir": str(cell.relative_to(root)).replace("\\", "/"),
                                  "bytes": sum(p.stat().st_size for p in cell.iterdir() if p.is_file()) if cell.is_dir() else 0,
                                  "status": "passed" if not errors else "failed", "errors": " | ".join(errors)})
    write_csv(output / "run_inventory_random_holdout_v1.csv", inventory)
    failures = [x for x in inventory if x["status"] != "passed"]
    if failures:
        write_json(output / "acceptance_random_holdout_v1.json",
                   {"status": "rejected", "failures": failures})
        raise RuntimeError(f"Raw run validation failed for {len(failures)} cells")
    write_csv(output / "individual_metrics_random_holdout_v1.csv", individual)

    ensemble_rows, ensemble_cache = [], {}
    for model in models:
        for split_id, record in split_records.items():
            total = np.zeros(data[split_id]["y"].shape, dtype=np.float64)
            for seed in seeds:
                total += np.load(result_root / model / split_id / f"seed_{seed}" / "probability.npy", mmap_mode="r")
            probability = total / len(seeds)
            ensemble_cache[(model, split_id)] = probability
            row = {"model_id": model, "split_id": split_id, "source_fold": record["source_fold"],
                   "seed_count": len(seeds)}
            for role in ("validation", "test"):
                for key, value in metrics(data[split_id]["y"], probability, data[split_id][role]).items():
                    row[f"{role}_{key}"] = value
            ensemble_rows.append(row)
    write_csv(output / "split_seed_ensemble_metrics_random_holdout_v1.csv", ensemble_rows)

    summaries = []
    for model in models:
        model_rows = [x for x in ensemble_rows if x["model_id"] == model]
        summary = {"model_id": model, "split_count": 4, "seed_count_per_split": 4}
        for name in METRICS:
            values = np.asarray([float(x[f"test_{name}"]) for x in model_rows])
            summary[f"split_ensemble_{name}_mean"] = float(values.mean())
            summary[f"split_ensemble_{name}_sd"] = float(values.std(ddof=1))
        raw_ap = np.asarray([float(x["test_average_precision"]) for x in individual if x["model_id"] == model])
        summary["individual_test_ap_mean"] = float(raw_ap.mean())
        summary["individual_test_ap_sd"] = float(raw_ap.std(ddof=1))
        summaries.append(summary)
    summaries.sort(key=lambda x: x["split_ensemble_average_precision_mean"], reverse=True)
    for rank, row in enumerate(summaries, 1):
        row["descriptive_rank"] = rank
    write_csv(output / "model_summary_random_holdout_v1.csv", summaries)

    spatial_rows = read_csv(root / "04_results/experiment_v4_aggregate/fold_seed_ensemble_metrics_v4.csv")
    spatial = {(x["model_id"], x["fold_id"]): x for x in spatial_rows}
    paired, model_comparison = [], []
    for model in models:
        deltas = []
        for split_id, record in split_records.items():
            random_row = next(x for x in ensemble_rows if x["model_id"] == model and x["split_id"] == split_id)
            spatial_row = spatial[(model, record["source_fold"])]
            random_ap, spatial_ap = float(random_row["test_average_precision"]), float(spatial_row["test_average_precision"])
            delta = random_ap - spatial_ap
            deltas.append(delta)
            paired.append({"model_id": model, "split_id": split_id, "source_fold": record["source_fold"],
                           "test_n": random_row["test_n"], "test_positive": random_row["test_positive"],
                           "random_test_ap": random_ap, "spatial_test_ap": spatial_ap,
                           "random_minus_spatial_ap": delta,
                           "random_to_spatial_ap_ratio": random_ap / spatial_ap,
                           "random_test_roc_auc": random_row["test_roc_auc"],
                           "spatial_test_roc_auc": spatial_row["test_roc_auc"]})
        random_values = np.asarray([float(x["test_average_precision"]) for x in ensemble_rows if x["model_id"] == model])
        spatial_values = np.asarray([float(spatial[(model, split_records[x["split_id"]]["source_fold"])]["test_average_precision"])
                                     for x in ensemble_rows if x["model_id"] == model])
        deltas = np.asarray(deltas)
        model_comparison.append({"model_id": model, "pairs": 4,
                                 "random_ap_mean": float(random_values.mean()), "random_ap_sd": float(random_values.std(ddof=1)),
                                 "spatial_ap_mean": float(spatial_values.mean()), "spatial_ap_sd": float(spatial_values.std(ddof=1)),
                                 "mean_random_minus_spatial_ap": float(deltas.mean()), "sd_paired_difference": float(deltas.std(ddof=1)),
                                 "random_higher_pairs": int((deltas > 0).sum())})
    write_csv(output / "spatial_random_paired_comparison_v1.csv", paired)
    write_csv(output / "spatial_random_model_summary_v1.csv", model_comparison)

    architecture_pairs = [("temporal_attn_vs_unet", "temporal_attn", "unet_matched"),
                          ("temporal_attn_vs_concat", "temporal_attn", "temporal_concat")]
    pair_rows, pair_summary = [], []
    ensemble_lookup = {(x["model_id"], x["split_id"]): x for x in ensemble_rows}
    for name, left, right in architecture_pairs:
        values = []
        for split_id in split_records:
            left_ap = float(ensemble_lookup[(left, split_id)]["test_average_precision"])
            right_ap = float(ensemble_lookup[(right, split_id)]["test_average_precision"])
            values.append(left_ap - right_ap)
            pair_rows.append({"comparison_id": name, "split_id": split_id, "left_model": left,
                              "right_model": right, "left_test_ap": left_ap, "right_test_ap": right_ap,
                              "ap_difference_left_minus_right": left_ap - right_ap})
        values = np.asarray(values)
        pair_summary.append({"comparison_id": name, "left_model": left, "right_model": right, "splits": 4,
                             "mean_ap_difference_left_minus_right": float(values.mean()),
                             "sd_ap_difference": float(values.std(ddof=1)),
                             "left_wins": int((values > 0).sum()), "right_wins": int((values < 0).sum())})
    write_csv(output / "architecture_pairwise_differences_random_v1.csv", pair_rows)
    write_csv(output / "architecture_pairwise_summary_random_v1.csv", pair_summary)

    acceptance = {"status": "accepted", "protocol_sha256": protocol_sha, "formal_cells": 48,
                  "validated_cells": len(inventory), "failed_cells": 0, "seed_ensembles": 12,
                  "maximum_metric_recalculation_delta": max_delta,
                  "spatial_comparator": {"status": "frozen_v4_2_with_disclosed_seam_exception",
                                         "aggregate_accepted": True, "independent_checks_passed": 792,
                                         "independent_checks_total": 793,
                                         "exception": "one of 37 severe V3 seam records improved by 49.51%, narrowly below the 50% target"},
                  "interpretation": "Random-pixel results are leakage-prone diagnostics and not spatial generalization estimates.",
                  "next_gate": "Independent recomputation of ensembles, paired differences, and all aggregate checksums."}
    write_json(output / "acceptance_random_holdout_v1.json", acceptance)
    (output / "README.md").write_text(
        "# random-holdout diagnostic V1\n\n"
        "This package validates 48 formal cells and compares four exact-count stratified random-pixel holdouts "
        "with their matched frozen V4.2 spatial folds. Random-pixel scores are leakage-prone diagnostics, not "
        "geographic generalization estimates.\n", encoding="utf-8", newline="\n")
    names = sorted(p.name for p in output.iterdir() if p.is_file() and p.name != "aggregate_checksums_random_v1.sha256")
    (output / "aggregate_checksums_random_v1.sha256").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="ascii", newline="\n")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
