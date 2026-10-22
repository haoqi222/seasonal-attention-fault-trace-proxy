#!/usr/bin/env python3
"""Independent recomputation of the V3 stage-9 aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROTOCOL_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def close(left, right, tolerance: float = 1e-12) -> bool:
    return math.isfinite(float(left)) and abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    audit_dir = root / "04_results/experiment_v3_aggregate"
    protocol_path = root / "03_configs/experiment_v3_protocol.json"
    protocol = read_json(protocol_path)
    result_root = root / "01_data/interim/results_v3" / PROTOCOL_SHA
    checks = 0
    failures: list[dict] = []

    def check(name: str, condition: bool, **details) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append({"check": name, **details})

    check("live_protocol_sha256", sha256(protocol_path) == PROTOCOL_SHA)
    required_audit = [
        "README.md",
        "aggregate_checksums_v3.sha256",
        "fold_seed_ensemble_metrics_v3.csv",
        "individual_metrics_v3.csv",
        "input_manifest_digest_v3.json",
        "model_summary_v3.csv",
        "pairwise_fold_ap_differences_v3.csv",
        "pairwise_summary_v3.csv",
        "pooled_oof_metrics_v3.csv",
        "run_inventory_v3.csv",
        "stage9_acceptance_v3.json",
        "training_diagnostics_v3.csv",
    ]
    for name in required_audit:
        check(f"audit_exists:{name}", (audit_dir / name).is_file())

    checksum_lines = (audit_dir / "aggregate_checksums_v3.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    checksum_map = dict(line.split("  ", 1)[::-1] for line in checksum_lines)
    check("aggregate_checksum_entry_count", len(checksum_map) == 11)
    for name, expected in checksum_map.items():
        check(f"aggregate_sha256:{name}", sha256(audit_dir / name) == expected)

    acceptance = read_json(audit_dir / "stage9_acceptance_v3.json")
    check("stage9_accepted", acceptance.get("overall_accepted") is True)
    check("stage9_validated_128", acceptance.get("validated_runs") == 128)
    check("stage9_failed_zero", acceptance.get("failed_runs") == 0)
    check("stage9_ensemble_32", acceptance.get("fold_seed_ensembles") == 32)
    check("stage9_model_summaries_8", acceptance.get("model_summaries") == 8)

    digest_record = read_json(audit_dir / "input_manifest_digest_v3.json")
    check("digest_protocol", digest_record.get("protocol_sha256") == PROTOCOL_SHA)
    for name, expected in digest_record.get("aggregation_source_sha256", {}).items():
        source_lookup = {
            "aggregation_script": root / "02_scripts/analysis/23_validate_and_aggregate_results_v3.py",
            "formal_runner": root / "02_scripts/modeling/13_run_experiment_v3.py",
            "scheduler": root / "02_scripts/modeling/14_schedule_experiments_v3.py",
            "dataset_loader": root / "02_scripts/modeling/dataset_common_v3.py",
            "protocol": protocol_path,
            "scheduler_execution_summary": root
            / "99_logs/experiment_v3"
            / PROTOCOL_SHA
            / "scheduler_execution_summary.json",
        }
        check(f"aggregation_source_sha256:{name}", sha256(source_lookup[name]) == expected)
    for relative, expected in digest_record.get("fold_input_sha256", {}).items():
        path = root / "01_data/interim/dataset_v3" / relative
        check(f"fold_input_sha256:{relative}", sha256(path) == expected)

    inventory = read_csv(audit_dir / "run_inventory_v3.csv")
    check("inventory_rows_128", len(inventory) == 128)
    check("inventory_unique_cells", len({row["cell_id"] for row in inventory}) == 128)
    check("inventory_all_passed", {row["status"] for row in inventory} == {"passed"})
    check("inventory_all_hashes_verified", {row["hashes_verified"] for row in inventory} == {"True"})
    check(
        "inventory_all_probabilities_verified",
        {row["probabilities_verified"] for row in inventory} == {"True"},
    )

    ordered_manifest_records: list[str] = []
    total_bytes = 0
    for row in inventory:
        run_dir = root / row["run_dir"]
        manifest_path = run_dir / "run_manifest.json"
        manifest_hash = sha256(manifest_path)
        check(f"manifest_sha256:{row['cell_id']}", manifest_hash == row["manifest_sha256"])
        manifest = read_json(manifest_path)
        check(f"manifest_completed:{row['cell_id']}", manifest.get("completed") is True)
        check(f"manifest_protocol:{row['cell_id']}", manifest.get("protocol_sha256") == PROTOCOL_SHA)
        for filename, expected in manifest.get("file_sha256", {}).items():
            check(
                f"run_file_sha256:{row['cell_id']}:{filename}",
                sha256(run_dir / filename) == expected,
            )
        current_bytes = sum(path.stat().st_size for path in run_dir.iterdir() if path.is_file())
        total_bytes += current_bytes
        check(f"run_bytes:{row['cell_id']}", current_bytes == int(row["bytes"]))
        ordered_manifest_records.append(f"{row['cell_id']} {manifest_hash}")
    ordered_digest = hashlib.sha256(
        ("\n".join(ordered_manifest_records) + "\n").encode("utf-8")
    ).hexdigest()
    check(
        "ordered_manifest_digest",
        ordered_digest == digest_record.get("ordered_manifest_digest_sha256"),
    )
    check("total_input_bytes", total_bytes == int(digest_record.get("total_input_bytes", -1)))

    folds = protocol["data"]["folds"]
    seeds = [int(seed) for seed in protocol["seeds"]]
    models = [item["model_id"] for item in protocol["deep_models"]] + [
        item["model_id"] for item in protocol["random_forest_models"]
    ]
    tiers = {
        item["model_id"]: item["analysis_tier"]
        for group in (protocol["deep_models"], protocol["random_forest_models"])
        for item in group
    }
    fold_arrays = {
        fold: {
            "y": np.load(root / "01_data/interim/dataset_v3" / fold / "y.npy", mmap_mode="r"),
            "validation": np.load(
                root / "01_data/interim/dataset_v3" / fold / "val.npy", mmap_mode="r"
            ).astype(bool),
            "test": np.load(
                root / "01_data/interim/dataset_v3" / fold / "test.npy", mmap_mode="r"
            ).astype(bool),
        }
        for fold in folds
    }
    stored_ensemble = {
        (row["model_id"], row["fold_id"]): row
        for row in read_csv(audit_dir / "fold_seed_ensemble_metrics_v3.csv")
    }
    check("stored_ensemble_rows_32", len(stored_ensemble) == 32)
    recomputed: dict[tuple[str, str], dict] = {}
    ensemble_scores: dict[tuple[str, str], np.ndarray] = {}
    for model_id in models:
        for fold_id in folds:
            data = fold_arrays[fold_id]
            for role, filename in (
                ("validation", "validation_probability.npy"),
                ("test", "test_probability.npy"),
            ):
                accumulator = np.zeros(data["y"].shape, dtype=np.float64)
                for seed in seeds:
                    path = result_root / model_id / fold_id / f"seed_{seed}" / filename
                    probability = np.load(path, mmap_mode="r")
                    check(f"probability_finite:{model_id}:{fold_id}:{seed}:{role}", np.isfinite(probability).all())
                    accumulator += probability
                ensemble = accumulator / 4.0
                role_metrics = metrics(data["y"], ensemble, data[role])
                recomputed[(model_id, fold_id, role)] = role_metrics
                if role == "test":
                    ensemble_scores[(model_id, fold_id)] = ensemble
                stored = stored_ensemble[(model_id, fold_id)]
                for key, value in role_metrics.items():
                    check(
                        f"ensemble_metric:{model_id}:{fold_id}:{role}:{key}",
                        close(stored[f"{role}_{key}"], value),
                    )

    model_summary = {row["model_id"]: row for row in read_csv(audit_dir / "model_summary_v3.csv")}
    pooled_rows = {row["model_id"]: row for row in read_csv(audit_dir / "pooled_oof_metrics_v3.csv")}
    check("model_summary_rows_8", len(model_summary) == 8)
    check("pooled_rows_8", len(pooled_rows) == 8)
    calculated_ap_means = {}
    for model_id in models:
        stored = model_summary[model_id]
        calculated_ap_means[model_id] = np.mean(
            [recomputed[(model_id, fold, "test")]["average_precision"] for fold in folds]
        )
        for metric_name in SUMMARY_METRICS:
            values = np.asarray(
                [recomputed[(model_id, fold, "test")][metric_name] for fold in folds]
            )
            check(
                f"model_mean:{model_id}:{metric_name}",
                close(stored[f"fold_ensemble_{metric_name}_mean"], values.mean()),
            )
            check(
                f"model_sd:{model_id}:{metric_name}",
                close(stored[f"fold_ensemble_{metric_name}_sd"], values.std(ddof=1)),
            )
        truth_parts = []
        score_parts = []
        for fold in folds:
            data = fold_arrays[fold]
            selected = data["test"] & (data["y"] != 2)
            truth_parts.append((data["y"][selected] == 1).astype(np.int8))
            score_parts.append(ensemble_scores[(model_id, fold)][selected])
        truth = np.concatenate(truth_parts)
        score = np.concatenate(score_parts)
        pooled = metrics(truth, score, np.ones(truth.shape, dtype=bool))
        for key, value in pooled.items():
            check(f"pooled_metric:{model_id}:{key}", close(pooled_rows[model_id][key], value))

    ranked = sorted(models, key=lambda model: calculated_ap_means[model], reverse=True)
    for rank, model_id in enumerate(ranked, 1):
        check(
            f"overall_rank:{model_id}",
            int(model_summary[model_id]["overall_descriptive_rank"]) == rank,
        )
    primary = [model for model in ranked if tiers[model] == "primary_capacity_matched"]
    for rank, model_id in enumerate(primary, 1):
        check(
            f"primary_rank:{model_id}",
            int(model_summary[model_id]["primary_capacity_matched_rank"]) == rank,
        )
    for model_id in models:
        if tiers[model_id] != "primary_capacity_matched":
            check(
                f"nonprimary_rank_blank:{model_id}",
                model_summary[model_id]["primary_capacity_matched_rank"] == "",
            )

    pairwise = read_csv(audit_dir / "pairwise_fold_ap_differences_v3.csv")
    pair_summary = {row["comparison_id"]: row for row in read_csv(audit_dir / "pairwise_summary_v3.csv")}
    check("pairwise_fold_rows_24", len(pairwise) == 24)
    check("pairwise_summary_rows_6", len(pair_summary) == 6)
    grouped: dict[str, list[float]] = {}
    for row in pairwise:
        difference = calculated = (
            recomputed[(row["left_model"], row["fold_id"], "test")]["average_precision"]
            - recomputed[(row["right_model"], row["fold_id"], "test")]["average_precision"]
        )
        check(
            f"pairwise_difference:{row['comparison_id']}:{row['fold_id']}",
            close(row["ap_difference_left_minus_right"], calculated),
        )
        grouped.setdefault(row["comparison_id"], []).append(difference)
    for comparison_id, differences in grouped.items():
        array = np.asarray(differences)
        stored = pair_summary[comparison_id]
        check(
            f"pairwise_mean:{comparison_id}",
            close(stored["mean_ap_difference_left_minus_right"], array.mean()),
        )
        check(
            f"pairwise_sd:{comparison_id}",
            close(stored["sd_ap_difference"], array.std(ddof=1)),
        )
        check(f"pairwise_left_wins:{comparison_id}", int(stored["left_wins"]) == int((array > 0).sum()))
        check(f"pairwise_right_wins:{comparison_id}", int(stored["right_wins"]) == int((array < 0).sum()))

    output = {
        "independent_verification_passed": not failures,
        "checks": checks,
        "failed_checks": len(failures),
        "failures": failures,
        "validated_runs": 128,
        "recomputed_fold_seed_ensembles": 32,
        "recomputed_models": 8,
        "protocol_sha256": PROTOCOL_SHA,
        "ordered_manifest_digest_sha256": ordered_digest,
        "descriptive_rank": ranked,
        "primary_capacity_matched_rank": primary,
    }
    with (audit_dir / "independent_verification_v3.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
