#!/usr/bin/env python3
"""Run isolated stage-6 smoke tests for the locked V3 runner.

Smoke artifacts are written under ``01_data/interim/smoke_test_v3`` and never
under formal ``results_v3``. Only F1 training patches are accessed. Validation
and test probability files are 1x1 zero schema sentinels, not predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import time
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


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("experiment_runner_v3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    runner_path = root / "02_scripts/modeling/13_run_experiment_v3.py"
    runner = load_runner(runner_path)
    context = runner.load_protocol_context(root)
    protocol_hash = context["protocol_sha256"]
    output_dir = (args.output_dir or root / "04_results/runner_smoke_v3").resolve()
    artifact_root = (
        args.artifact_root
        or root / "01_data/interim/smoke_test_v3" / protocol_hash
    ).resolve()
    for target in (output_dir, artifact_root):
        if target.exists():
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite smoke target: {target}")
            if target == root or root not in target.parents:
                raise ValueError(f"Unsafe smoke target for removal: {target}")
            shutil.rmtree(target)
        target.mkdir(parents=True)

    formal_results = root / "01_data/interim/results_v3"
    formal_absent_before = not formal_results.exists()
    if not formal_absent_before:
        raise FileExistsError("Formal results_v3 exists before stage-6 smoke testing")

    fold_id = "F1_NorthTest"
    source_paths = {
        "protocol": root / "03_configs/experiment_v3_protocol.json",
        "stage5_lock": root / "04_results/training_protocol_v3/protocol_lock_v3.json",
        "stage5_acceptance": root / "04_results/training_protocol_v3/stage5_acceptance_v3.json",
        "runner": runner_path,
        "loader": root / "02_scripts/modeling/dataset_common_v3.py",
        "models": root / "02_scripts/modeling/models_v3.py",
        "temporal_cmt": root / "02_scripts/modeling/temporal_cmt.py",
        "X": root / "01_data/interim/dataset_v2/X.npy",
        "F1_y": root / f"01_data/interim/dataset_v3/{fold_id}/y.npy",
        "F1_partition": root / f"01_data/interim/dataset_v3/{fold_id}/partition.npy",
        "F1_train": root / f"01_data/interim/dataset_v3/{fold_id}/train.npy",
        "F1_normalization": root / f"01_data/interim/normalization_v3/{fold_id}/norm_stats.json",
    }
    hashes_before = {name: sha256(path) for name, path in source_paths.items()}
    data = runner.load_fold(fold_id, project_root=root, normalize=True, verify_hashes=True)

    summary_rows = []
    deep_specs = [
        ("cpu_unet_matched", "unet_matched", "cpu", 1),
        ("gpu0_unet_matched", "unet_matched", "cuda:0", 16),
        ("gpu1_temporal_attn", "temporal_attn", "cuda:1", 16),
        ("gpu0_temporal_concat", "temporal_concat", "cuda:0", 16),
        ("gpu1_summer_only", "summer_only", "cuda:1", 16),
        ("gpu0_thermal_only", "thermal_only", "cuda:0", 16),
        ("gpu1_optical_only", "optical_only", "cuda:1", 16),
    ]
    for smoke_id, model_id, device, batch_size in deep_specs:
        started = time.time()
        run_dir = artifact_root / smoke_id
        result = runner.smoke_deep_step(
            context,
            data=data,
            model_id=model_id,
            fold_id=fold_id,
            seed=0,
            device=device,
            batch_size=batch_size,
            run_dir=run_dir,
        )
        summary_rows.append(
            {
                "smoke_id": smoke_id,
                "model_id": model_id,
                "model_kind": "deep",
                "device": device,
                "batch_size": batch_size,
                "loss": format(result["loss"], ".17g"),
                "gradient_norm": format(result["gradient_norm_before_clip"], ".17g"),
                "feature_count": "",
                "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
                "roundtrip_equal": result["checkpoint_roundtrip_equal"],
                "no_validation_or_test_access": result["smoke_no_validation_or_test_access"],
                "seconds": format(time.time() - started, ".6f"),
                "status": "passed",
                "artifact_directory": str(run_dir.relative_to(root)),
            }
        )

    for model_id in runner.RF_MODEL_IDS:
        started = time.time()
        run_dir = artifact_root / f"cpu_{model_id}"
        result = runner.smoke_rf_step(
            context,
            data=data,
            model_id=model_id,
            fold_id=fold_id,
            seed=0,
            run_dir=run_dir,
        )
        summary_rows.append(
            {
                "smoke_id": f"cpu_{model_id}",
                "model_id": model_id,
                "model_kind": "random_forest",
                "device": "cpu",
                "batch_size": "",
                "loss": "",
                "gradient_norm": "",
                "feature_count": result["feature_count"],
                "peak_cuda_memory_bytes": 0,
                "roundtrip_equal": result["model_roundtrip_equal"],
                "no_validation_or_test_access": result["smoke_no_validation_or_test_access"],
                "seconds": format(time.time() - started, ".6f"),
                "status": "passed",
                "artifact_directory": str(run_dir.relative_to(root)),
            }
        )

    required_common = set(context["protocol"]["output_schema"]["required_run_files"])
    checks = []

    def record(check: str, passed: bool, **values) -> None:
        checks.append({"check": check, "passed": bool(passed), **values})

    record("formal_results_absent_before_smoke", formal_absent_before)
    record("nine_smoke_cells_completed", len(summary_rows) == 9, count=len(summary_rows))
    record(
        "all_smoke_cells_passed",
        all(row["status"] == "passed" for row in summary_rows),
    )
    record(
        "cpu_and_both_GPUs_exercised",
        {row["device"] for row in summary_rows} >= {"cpu", "cuda:0", "cuda:1"},
    )
    record(
        "all_six_deep_designs_exercised_on_GPU_batch16",
        {
            row["model_id"]
            for row in summary_rows
            if row["model_kind"] == "deep"
            and row["device"].startswith("cuda")
            and row["batch_size"] == 16
        }
        == set(runner.DEEP_MODEL_IDS),
    )
    record(
        "both_RF_feature_designs_exercised",
        {
            row["model_id"]
            for row in summary_rows
            if row["model_kind"] == "random_forest"
        }
        == set(runner.RF_MODEL_IDS),
    )
    record(
        "all_roundtrips_equal",
        all(bool(row["roundtrip_equal"]) for row in summary_rows),
    )
    record(
        "all_smoke_cells_avoid_validation_and_test",
        all(bool(row["no_validation_or_test_access"]) for row in summary_rows),
    )

    artifact_records = []
    for row in summary_rows:
        run_dir = root / row["artifact_directory"]
        actual_names = {path.name for path in run_dir.iterdir() if path.is_file()}
        kind_specific = "checkpoint.pt" if row["model_kind"] == "deep" else "model.joblib"
        schema_ok = required_common <= actual_names and kind_specific in actual_names
        sentinel_ok = bool(
            np.array_equal(
                np.load(run_dir / "validation_probability.npy"),
                np.zeros((1, 1), dtype=np.float32),
            )
            and np.array_equal(
                np.load(run_dir / "test_probability.npy"),
                np.zeros((1, 1), dtype=np.float32),
            )
        )
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        metadata_ok = bool(
            result["scope"] == "stage6_smoke_only"
            and result["smoke_no_validation_or_test_access"]
            and not result["formal_training_started"]
            and manifest["smoke_no_validation_or_test_access"]
            and manifest["protocol_sha256"] == protocol_hash
        )
        record(f"{row['smoke_id']}_output_schema", schema_ok)
        record(f"{row['smoke_id']}_zero_role_sentinels", sentinel_ok)
        record(f"{row['smoke_id']}_smoke_metadata", metadata_ok)
        artifact_records.append(
            {
                "smoke_id": row["smoke_id"],
                "directory": row["artifact_directory"],
                "files": {
                    path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in sorted(run_dir.iterdir())
                    if path.is_file()
                },
            }
        )

    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    sources_unchanged = hashes_before == hashes_after
    formal_absent_after = not formal_results.exists()
    record("all_smoke_sources_unchanged", sources_unchanged)
    record("formal_results_absent_after_smoke", formal_absent_after)
    record(
        "protocol_hash_unchanged",
        sha256(context["protocol_path"]) == protocol_hash,
        protocol_sha256=protocol_hash,
    )

    write_csv(output_dir / "smoke_summary_v3.csv", summary_rows)
    write_json(
        output_dir / "smoke_artifact_manifest_v3.json",
        {
            "protocol_sha256": protocol_hash,
            "artifact_root": str(artifact_root.relative_to(root)),
            "formal_results_root": "01_data/interim/results_v3",
            "formal_results_absent_before": formal_absent_before,
            "formal_results_absent_after": formal_absent_after,
            "artifacts": artifact_records,
        },
    )
    write_json(
        output_dir / "smoke_source_integrity_v3.json",
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
        output_dir / "smoke_environment_v3.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
    )
    overall_accepted = all(item["passed"] for item in checks)
    write_json(
        output_dir / "stage6_acceptance_v3.json",
        {
            "overall_accepted": overall_accepted,
            "status": (
                "accepted_for_formal_run_orchestration"
                if overall_accepted else "rejected"
            ),
            "next_gate": (
                "Create a resumable scheduler for the 128 locked cells, perform a preflight-only "
                "matrix audit, then request explicit authorization before launching formal training."
            ),
            "checks": checks,
            "formal_training_started": False,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# V3 runner smoke test — stage 6",
                "",
                f"Status: {'accepted' if overall_accepted else 'rejected'}.",
                "",
                f"Protocol SHA-256: `{protocol_hash}`",
                "",
                "The smoke suite exercised one CPU deep batch, all six deep designs at the frozen",
                "GPU batch size across both RTX 5090 devices, and both RF feature families. It used",
                "F1 training pixels only. Formal results_v3 was absent before and after the suite.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {
            "runner_smoke_checksums_v3.sha256",
            "independent_verification_v3.json",
        }:
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "runner_smoke_checksums_v3.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "artifact_root": str(artifact_root),
                "overall_accepted": overall_accepted,
                "checks": len(checks),
                "failed_checks": sum(not item["passed"] for item in checks),
                "smoke_cells": len(summary_rows),
                "formal_training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
