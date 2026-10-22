#!/usr/bin/env python3
"""Audit and lock the V3 training protocol before any training.

The script validates model capacity, forward shapes, fold-specific weights and
sampler fractions, the complete run matrix, environment readiness, output and
failure rules, and source hashes. It does not create an optimizer, execute
backpropagation, access a test prediction, or create ``results_v3``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    protocol_path = (
        args.protocol or root / "03_configs/experiment_v3_protocol.json"
    ).resolve()
    output_dir = (
        args.output_dir or root / "04_results/training_protocol_v3"
    ).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        if output_dir == root or root not in output_dir.parents:
            raise ValueError(f"Unsafe output directory for removal: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    modeling_dir = root / "02_scripts/modeling"
    sys.path.insert(0, str(modeling_dir))
    from models_v3 import (  # pylint: disable=import-outside-toplevel
        DEEP_MODEL_IDS,
        build_deep_model,
        trainable_parameters,
    )

    source_paths = {
        "protocol": protocol_path,
        "models_v3": modeling_dir / "models_v3.py",
        "temporal_cmt": modeling_dir / "temporal_cmt.py",
        "cross_modal": modeling_dir / "cross_modal.py",
        "dataset_common_v3": modeling_dir / "dataset_common_v3.py",
        "fold_manifest": root / "04_results/split_design_v3/fold_manifest_v3.json",
        "stage2_acceptance": root / "01_data/interim/dataset_v3/stage2_acceptance_v3.json",
        "stage3_acceptance": root / "01_data/interim/normalization_v3/stage3_acceptance_v3.json",
        "stage4_acceptance": root / "04_results/loader_audit_v3/stage4_acceptance_v3.json",
        "loader_summary": root / "04_results/loader_audit_v3/loader_audit_summary_v3.csv",
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protocol source {name}: {path}")
    hashes_before = {name: sha256(path) for name, path in source_paths.items()}
    protocol_sha256 = hashes_before["protocol"]

    checks: list[dict] = []

    def record(check: str, passed: bool, **values) -> None:
        checks.append({"check": check, "passed": bool(passed), **values})

    record(
        "protocol_status_frozen_before_training",
        protocol.get("status") == "frozen_before_training",
    )
    acceptance_sources = {
        key: json.loads(source_paths[key].read_text(encoding="utf-8"))
        for key in ("stage2_acceptance", "stage3_acceptance", "stage4_acceptance")
    }
    for key, payload in acceptance_sources.items():
        record(
            f"{key}_clean",
            payload.get("overall_accepted")
            and all(item["passed"] for item in payload["checks"]),
        )

    fold_manifest = json.loads(source_paths["fold_manifest"].read_text(encoding="utf-8"))
    manifest_folds = [record_["fold_id"] for record_ in fold_manifest["folds"]]
    folds = protocol["data"]["folds"]
    seeds = protocol["seeds"]
    record("protocol_folds_match_frozen_manifest", folds == manifest_folds, folds=folds)
    record("four_unique_seeds_frozen", seeds == [0, 1, 2, 3], seeds=seeds)

    configured_deep_ids = [entry["model_id"] for entry in protocol["deep_models"]]
    record(
        "configured_deep_models_match_factory",
        tuple(configured_deep_ids) == DEEP_MODEL_IDS,
        model_ids=configured_deep_ids,
    )
    model_rows: list[dict] = []
    torch.manual_seed(0)
    dummy = torch.zeros(1, 32, protocol["data"]["patch_pixels"], protocol["data"]["patch_pixels"])
    for entry in protocol["deep_models"]:
        model_id = entry["model_id"]
        model = build_deep_model(model_id).eval()
        actual_parameters = trainable_parameters(model)
        with torch.no_grad():
            output = model(dummy)
        parameter_match = actual_parameters == entry["expected_trainable_parameters"]
        forward_valid = tuple(output.shape) == (1, 1, 128, 128) and bool(
            torch.isfinite(output).all()
        )
        record(
            f"model_{model_id}_parameter_count",
            parameter_match,
            actual=actual_parameters,
            expected=entry["expected_trainable_parameters"],
        )
        record(
            f"model_{model_id}_forward_shape_and_finiteness",
            forward_valid,
            output_shape=list(output.shape),
        )
        model_rows.append(
            {
                "model_id": model_id,
                "analysis_tier": entry["analysis_tier"],
                "input": entry["input"],
                "trainable_parameters": actual_parameters,
                "parameters_M": format(actual_parameters / 1e6, ".9f"),
                "forward_shape": "1x1x128x128",
                "forward_finite": forward_valid,
            }
        )
        del model, output
    primary_counts = [
        row["trainable_parameters"]
        for row in model_rows
        if row["analysis_tier"] == "primary_capacity_matched"
    ]
    capacity_ratio = max(primary_counts) / min(primary_counts)
    record(
        "primary_deep_capacity_ratio_at_most_1_5",
        len(primary_counts) == 3 and capacity_ratio <= 1.5,
        ratio=capacity_ratio,
    )
    record(
        "legacy_large_resnet_unet_excluded_from_v3_primary",
        "unet_resnet18" not in configured_deep_ids
        and max(primary_counts) < 1_000_000,
    )

    with source_paths["loader_summary"].open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        loader_rows = {row["fold_id"]: row for row in csv.DictReader(handle)}
    configured_fractions = protocol["deep_training"]["sampler"][
        "fold_fault_center_fraction"
    ]
    configured_weights = protocol["deep_training"]["loss"]["fold_positive_weight"]
    for fold_id in folds:
        row = loader_rows[fold_id]
        positive_centers = int(row["positive_training_centers"])
        background_centers = int(row["background_training_centers"])
        natural_fraction = positive_centers / (positive_centers + background_centers)
        pos_weight = int(row["negative_training_pixels"]) / int(
            row["positive_training_pixels"]
        )
        record(
            f"{fold_id}_natural_center_fraction_matches_loader",
            np.isclose(
                configured_fractions[fold_id], natural_fraction, rtol=0.0, atol=1e-15
            ),
            configured=configured_fractions[fold_id],
            recomputed=natural_fraction,
        )
        record(
            f"{fold_id}_positive_weight_matches_loader",
            np.isclose(configured_weights[fold_id], pos_weight, rtol=0.0, atol=1e-15),
            configured=configured_weights[fold_id],
            recomputed=pos_weight,
        )

    deep_training = protocol["deep_training"]
    record(
        "steps_per_epoch_consistent",
        deep_training["patches_per_epoch"] // deep_training["batch_size"]
        == deep_training["steps_per_epoch"]
        and deep_training["patches_per_epoch"] % deep_training["batch_size"] == 0,
    )
    record(
        "early_stop_has_minimum_training_horizon",
        deep_training["minimum_epochs_before_early_stop"] >= 15
        and deep_training["checkpoint"]["patience_epochs"] >= 10
        and deep_training["maximum_epochs"] >= 60,
    )
    record(
        "checkpoint_uses_threshold_free_validation_AP",
        deep_training["checkpoint"]["monitor"] == "validation_average_precision"
        and not deep_training["checkpoint"]["test_data_used"],
    )
    record(
        "same_optimizer_schedule_for_all_deep_models",
        deep_training["schedule"]["same_for_every_deep_model"]
        and deep_training["optimizer"]["name"] == "AdamW",
    )
    record(
        "primary_evaluation_has_no_TTA_or_test_tuning",
        not deep_training["primary_test_time_augmentation"]
        and not protocol["evaluation"]["test_threshold_tuning"],
    )
    record(
        "primary_metric_and_summary_are_frozen",
        protocol["evaluation"]["primary_test_metric"] == "average precision"
        and "equal-weight mean" in protocol["evaluation"]["primary_model_summary"],
    )
    record(
        "random_forest_feature_fairness_is_explicit",
        protocol["random_forest_models"][0]["model_id"] == "rf_direct"
        and protocol["random_forest_models"][0]["analysis_tier"]
        == "primary_nonneural_baseline"
        and protocol["random_forest_models"][1]["analysis_tier"]
        == "secondary_context_sensitivity",
    )
    record(
        "failure_policy_forbids_seed_replacement_and_one_off_OOM_fix",
        protocol["failure_policy"]["failed_seed_replacement"] == "prohibited"
        and "do not use a one-off batch size"
        in protocol["failure_policy"]["out_of_memory"],
    )
    record(
        "change_control_forbids_cross_protocol_mixing",
        protocol["change_control"]["mixing_results_across_protocol_hashes"]
        == "prohibited"
        and protocol["change_control"]["test_driven_protocol_change"] == "prohibited",
    )

    all_models = [
        (entry["model_id"], "deep", entry["analysis_tier"])
        for entry in protocol["deep_models"]
    ] + [
        (entry["model_id"], "random_forest", entry["analysis_tier"])
        for entry in protocol["random_forest_models"]
    ]
    run_rows: list[dict] = []
    for model_id, kind, tier in all_models:
        for fold_id in folds:
            for seed in seeds:
                run_rows.append(
                    {
                        "run_id": f"{model_id}__{fold_id}__seed{seed}",
                        "model_id": model_id,
                        "model_kind": kind,
                        "analysis_tier": tier,
                        "fold_id": fold_id,
                        "seed": seed,
                        "status": "planned_not_started",
                        "output_directory": (
                            f"01_data/interim/results_v3/{protocol_sha256}/"
                            f"{model_id}/{fold_id}/seed_{seed}"
                        ),
                    }
                )
    deep_runs = sum(row["model_kind"] == "deep" for row in run_rows)
    rf_runs = sum(row["model_kind"] == "random_forest" for row in run_rows)
    record(
        "run_matrix_counts_match_protocol",
        deep_runs == protocol["run_matrix"]["deep_runs"]
        and rf_runs == protocol["run_matrix"]["random_forest_runs"]
        and len(run_rows) == protocol["run_matrix"]["total_runs"],
        deep_runs=deep_runs,
        random_forest_runs=rf_runs,
        total_runs=len(run_rows),
    )
    record(
        "run_ids_and_output_directories_unique",
        len({row["run_id"] for row in run_rows}) == len(run_rows)
        and len({row["output_directory"] for row in run_rows}) == len(run_rows),
    )
    results_v3 = root / "01_data/interim/results_v3"
    record(
        "results_v3_absent_before_protocol_freeze",
        not results_v3.exists(),
        path=str(results_v3.relative_to(root)),
    )

    gpu_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    gpu_records = []
    for line in gpu_lines:
        index, name, memory_mib, driver = [part.strip() for part in line.split(",", 3)]
        gpu_records.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "driver": driver,
            }
        )
    disk = shutil.disk_usage(root)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "torch": torch.__version__,
            "segmentation-models-pytorch": package_version(
                "segmentation-models-pytorch"
            ),
        },
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "gpus": gpu_records,
        "project_disk_total_bytes": disk.total,
        "project_disk_free_bytes": disk.free,
    }
    record(
        "two_RTX_5090_GPUs_ready",
        torch.cuda.is_available()
        and torch.cuda.device_count() == 2
        and len(gpu_records) == 2
        and all("RTX 5090" in gpu["name"] for gpu in gpu_records)
        and all(gpu["memory_mib"] >= 32000 for gpu in gpu_records),
    )
    record("at_least_500GB_project_disk_free", disk.free >= 500 * 1024 ** 3)
    write_json(output_dir / "environment_snapshot_v3.json", environment)
    write_csv(output_dir / "model_capacity_v3.csv", model_rows)
    write_csv(output_dir / "run_matrix_v3.csv", run_rows)

    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    sources_unchanged = hashes_before == hashes_after
    record("all_protocol_sources_unchanged_during_stage5", sources_unchanged)
    overall_accepted = all(item["passed"] for item in checks)
    write_json(
        output_dir / "protocol_lock_v3.json",
        {
            "schema_version": "fault-proxy-v3-protocol-lock-1.0",
            "status": "locked" if overall_accepted else "rejected",
            "protocol_path": str(protocol_path.relative_to(root)),
            "protocol_sha256": protocol_sha256,
            "source_sha256": hashes_before,
            "model_capacity_ratio_primary": capacity_ratio,
            "planned_runs": {
                "deep": deep_runs,
                "random_forest": rf_runs,
                "total": len(run_rows),
            },
            "training_started": False,
        },
    )
    write_json(
        output_dir / "protocol_source_integrity_v3.json",
        {
            "source_files": {
                name: {
                    "path": str(path.relative_to(root)),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                }
                for name, path in source_paths.items()
            },
            "all_sources_unchanged": sources_unchanged,
        },
    )
    write_json(
        output_dir / "stage5_acceptance_v3.json",
        {
            "overall_accepted": overall_accepted,
            "status": (
                "accepted_for_runner_implementation_and_smoke_test"
                if overall_accepted else "rejected"
            ),
            "next_gate": (
                "Implement a protocol-hash-bound runner and pass CPU/GPU one-batch smoke tests, "
                "checkpoint round-trip, and output-schema validation before full training."
            ),
            "checks": checks,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# V3 fair training protocol — stage 5",
                "",
                f"Status: {'locked and accepted' if overall_accepted else 'rejected'}.",
                "",
                f"Protocol SHA-256: `{protocol_sha256}`",
                "",
                "The protocol freezes eight designs, four spatial folds, four seeds, common deep",
                "optimization and stopping rules, fold-specific weights, evaluation aggregation,",
                "failure handling, and the required output schema. The 128 planned cells remain",
                "`planned_not_started`; this stage creates no training results or checkpoints.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {
            "training_protocol_checksums_v3.sha256",
            "independent_verification_v3.json",
        }:
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "training_protocol_checksums_v3.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overall_accepted": overall_accepted,
                "protocol_sha256": protocol_sha256,
                "checks": len(checks),
                "failed_checks": sum(not item["passed"] for item in checks),
                "planned_runs": len(run_rows),
                "training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
