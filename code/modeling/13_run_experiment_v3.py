#!/usr/bin/env python3
"""Protocol-bound runner for the V3 experiment matrix.

Formal execution requires the exact frozen protocol SHA-256 on the command
line. The module also exposes isolated one-step smoke helpers used by stage 6.
Smoke helpers use training pixels only and write outside ``results_v3``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common_v3 import (  # noqa: E402
    FoldTrainPatchDataset,
    best_threshold,
    inference_window_starts,
    load_fold,
    rf_features,
    sliding_predict_fold,
)
from models_v3 import DEEP_MODEL_IDS, build_deep_model, trainable_parameters  # noqa: E402


RF_MODEL_IDS = ("rf_direct", "rf_context")


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
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_protocol_context(project_root: Path | str) -> dict:
    root = Path(project_root).resolve()
    protocol_path = root / "03_configs/experiment_v3_protocol.json"
    lock_path = root / "04_results/training_protocol_v3/protocol_lock_v3.json"
    acceptance_path = root / "04_results/training_protocol_v3/stage5_acceptance_v3.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    if lock["status"] != "locked" or lock["protocol_sha256"] != protocol_hash:
        raise ValueError("Stage-5 protocol lock does not match the live protocol")
    if lock["training_started"]:
        raise ValueError("Stage-5 lock unexpectedly says training already started")
    if not acceptance["overall_accepted"] or not all(
        item["passed"] for item in acceptance["checks"]
    ):
        raise ValueError("Stage-5 protocol acceptance is not clean")
    return {
        "root": root,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_hash,
        "lock": lock,
    }


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def environment_record(device: str) -> dict:
    record = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
    }
    if device.startswith("cuda"):
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        record["gpu_index"] = index
        record["gpu_name"] = torch.cuda.get_device_name(index)
        record["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(index).total_memory
    return record


def deep_optimizer_scheduler(model: nn.Module, protocol: dict):
    training = protocol["deep_training"]
    optimizer_cfg = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_cfg["learning_rate"],
        weight_decay=optimizer_cfg["weight_decay"],
    )
    maximum_epochs = training["maximum_epochs"]
    warmup_epochs = training["schedule"]["warmup_epochs"]
    base_lr = optimizer_cfg["learning_rate"]
    minimum_factor = training["schedule"]["minimum_learning_rate"] / base_lr

    def multiplier(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return (epoch_index + 1) / warmup_epochs
        denominator = max(maximum_epochs - warmup_epochs - 1, 1)
        progress = min(max((epoch_index - warmup_epochs) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return optimizer, scheduler


def evaluation_metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(probability[selected], 0.0, 1.0)
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


def save_pr_curve(path: Path, y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> None:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(probability[selected], 0.0, 1.0)
    precision, recall, thresholds = precision_recall_curve(truth, score)
    np.savez_compressed(path, precision=precision, recall=recall, thresholds=thresholds)


def code_hashes(root: Path) -> dict:
    paths = {
        "runner": root / "02_scripts/modeling/13_run_experiment_v3.py",
        "loader": root / "02_scripts/modeling/dataset_common_v3.py",
        "models": root / "02_scripts/modeling/models_v3.py",
        "temporal_cmt": root / "02_scripts/modeling/temporal_cmt.py",
        "cross_modal": root / "02_scripts/modeling/cross_modal.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def finalize_manifest(run_dir: Path, metadata: dict) -> None:
    file_hashes = {
        path.name: sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    write_json(run_dir / "run_manifest.json", {**metadata, "file_sha256": file_hashes})


def write_smoke_schema_files(
    run_dir: Path,
    *,
    context: dict,
    model_id: str,
    fold_id: str,
    seed: int,
    device: str,
    kind: str,
    history: dict,
    result: dict,
) -> None:
    protocol = context["protocol"]
    write_json(
        run_dir / "config_snapshot.json",
        {
            "scope": "stage6_smoke_only",
            "protocol_sha256": context["protocol_sha256"],
            "model_id": model_id,
            "fold_id": fold_id,
            "seed": seed,
            "device": device,
            "formal_protocol_unchanged": True,
        },
    )
    write_json(run_dir / "environment.json", environment_record(device))
    write_csv(run_dir / "history.csv", [history])
    write_json(run_dir / "result.json", result)
    np.save(run_dir / "validation_probability.npy", np.zeros((1, 1), dtype=np.float32))
    np.save(run_dir / "test_probability.npy", np.zeros((1, 1), dtype=np.float32))
    np.savez_compressed(
        run_dir / "pr_curve.npz",
        precision=np.array([], dtype=np.float64),
        recall=np.array([], dtype=np.float64),
        thresholds=np.array([], dtype=np.float64),
        smoke_no_role_access=np.array([True]),
    )
    finalize_manifest(
        run_dir,
        {
            "scope": "stage6_smoke_only",
            "model_kind": kind,
            "model_id": model_id,
            "fold_id": fold_id,
            "seed": seed,
            "protocol_sha256": context["protocol_sha256"],
            "code_sha256": code_hashes(context["root"]),
            "formal_required_files": protocol["output_schema"]["required_run_files"],
            "smoke_no_validation_or_test_access": True,
        },
    )


def smoke_deep_step(
    context: dict,
    *,
    data: dict,
    model_id: str,
    fold_id: str,
    seed: int,
    device: str,
    batch_size: int,
    run_dir: Path,
) -> dict:
    """Execute one training-only batch and checkpoint round-trip."""

    set_determinism(seed)
    protocol = context["protocol"]
    fraction = protocol["deep_training"]["sampler"]["fold_fault_center_fraction"][fold_id]
    dataset = FoldTrainPatchDataset(
        data, fault_frac=fraction, n=batch_size, seed=seed, patch=protocol["data"]["patch_pixels"]
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    x, y, mask = next(iter(loader))
    target_device = torch.device(device)
    if device.startswith("cuda"):
        torch.cuda.set_device(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    model = build_deep_model(model_id).to(target_device)
    optimizer, scheduler = deep_optimizer_scheduler(model, protocol)
    positive_weight = protocol["deep_training"]["loss"]["fold_positive_weight"][fold_id]
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=target_device)
    )
    x, y, mask = x.to(target_device), y.to(target_device), mask.to(target_device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x).squeeze(1)
    selected = mask > 0
    if not selected.any():
        raise RuntimeError("Smoke batch has an empty loss mask")
    loss = criterion(output[selected], y[selected].float())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {model_id} {device}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), protocol["deep_training"]["optimizer"]["gradient_clip_norm"]
    )
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite gradient norm: {model_id} {device}")
    optimizer.step()
    scheduler.step()

    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint = {
        "scope": "stage6_smoke_only",
        "protocol_sha256": context["protocol_sha256"],
        "model_id": model_id,
        "fold_id": fold_id,
        "seed": seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "smoke_no_validation_or_test_access": True,
    }
    torch.save(checkpoint, checkpoint_path)

    restored_model = build_deep_model(model_id).to(target_device)
    restored_optimizer, restored_scheduler = deep_optimizer_scheduler(restored_model, protocol)
    restored = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if restored["protocol_sha256"] != context["protocol_sha256"]:
        raise RuntimeError("Checkpoint protocol hash mismatch")
    restored_model.load_state_dict(restored["model_state"])
    restored_optimizer.load_state_dict(restored["optimizer_state"])
    restored_scheduler.load_state_dict(restored["scheduler_state"])
    model.eval()
    restored_model.eval()
    with torch.no_grad():
        reference = model(x[:1])
        reloaded = restored_model(x[:1])
    roundtrip_equal = bool(torch.equal(reference, reloaded))
    if not roundtrip_equal:
        raise RuntimeError(f"Checkpoint round-trip changed output: {model_id}")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(target_device))
        if device.startswith("cuda") else 0
    )
    result = {
        "scope": "stage6_smoke_only",
        "model_id": model_id,
        "fold_id": fold_id,
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "trainable_parameters": trainable_parameters(model),
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
        "checkpoint_roundtrip_equal": roundtrip_equal,
        "peak_cuda_memory_bytes": peak_memory,
        "smoke_no_validation_or_test_access": True,
        "formal_training_started": False,
    }
    write_smoke_schema_files(
        run_dir,
        context=context,
        model_id=model_id,
        fold_id=fold_id,
        seed=seed,
        device=device,
        kind="deep",
        history={
            "smoke_step": 1,
            "loss": result["loss"],
            "gradient_norm_before_clip": result["gradient_norm_before_clip"],
            "learning_rate_after_step": optimizer.param_groups[0]["lr"],
        },
        result=result,
    )
    del model, restored_model, optimizer, restored_optimizer, x, y, mask, output
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def smoke_rf_step(
    context: dict,
    *,
    data: dict,
    model_id: str,
    fold_id: str,
    seed: int,
    run_dir: Path,
    samples_per_class: int = 2048,
) -> dict:
    """Fit and serialize a small training-only RF smoke model."""

    if model_id not in RF_MODEL_IDS:
        raise ValueError(model_id)
    rng = np.random.default_rng(seed)
    positive = np.argwhere(data["train"] & (data["y"] == 1))
    background = np.argwhere(data["train"] & (data["y"] == 0))
    positive = positive[
        rng.choice(len(positive), min(samples_per_class, len(positive)), replace=False)
    ]
    background = background[
        rng.choice(len(background), min(samples_per_class, len(background)), replace=False)
    ]
    feature_cube = data["X"] if model_id == "rf_direct" else rf_features(data)
    features = np.vstack(
        [
            feature_cube[positive[:, 0], positive[:, 1]],
            feature_cube[background[:, 0], background[:, 1]],
        ]
    )
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int8), np.zeros(len(background), dtype=np.int8)]
    )
    classifier = RandomForestClassifier(
        n_estimators=5,
        min_samples_leaf=context["protocol"]["random_forest_training"]["min_samples_leaf"],
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=seed,
    )
    classifier.fit(features, labels)
    run_dir.mkdir(parents=True)
    model_path = run_dir / "model.joblib"
    joblib.dump(classifier, model_path)
    restored = joblib.load(model_path)
    probe = features[:64]
    roundtrip_equal = bool(
        np.array_equal(classifier.predict_proba(probe), restored.predict_proba(probe))
    )
    if not roundtrip_equal:
        raise RuntimeError(f"RF serialization round-trip failed: {model_id}")
    result = {
        "scope": "stage6_smoke_only",
        "model_id": model_id,
        "fold_id": fold_id,
        "seed": seed,
        "device": "cpu",
        "smoke_estimators": 5,
        "formal_estimators": context["protocol"]["random_forest_training"]["n_estimators"],
        "training_samples": int(len(labels)),
        "feature_count": int(features.shape[1]),
        "model_roundtrip_equal": roundtrip_equal,
        "smoke_no_validation_or_test_access": True,
        "formal_training_started": False,
    }
    write_smoke_schema_files(
        run_dir,
        context=context,
        model_id=model_id,
        fold_id=fold_id,
        seed=seed,
        device="cpu",
        kind="random_forest",
        history={
            "smoke_step": 1,
            "smoke_estimators": 5,
            "training_samples": len(labels),
            "feature_count": features.shape[1],
        },
        result=result,
    )
    if model_id == "rf_context":
        del feature_cube
    return result


def formal_output_directory(context: dict, model_id: str, fold_id: str, seed: int) -> Path:
    return (
        context["root"]
        / "01_data/interim/results_v3"
        / context["protocol_sha256"]
        / model_id
        / fold_id
        / f"seed_{seed}"
    )


def validate_formal_cell(context: dict, model_id: str, fold_id: str, seed: int) -> str:
    protocol = context["protocol"]
    if fold_id not in protocol["data"]["folds"]:
        raise ValueError(f"Fold is not in the locked protocol: {fold_id}")
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed is not in the locked protocol: {seed}")
    if model_id in DEEP_MODEL_IDS:
        return "deep"
    if model_id in RF_MODEL_IDS:
        return "random_forest"
    raise ValueError(f"Model is not in the locked protocol: {model_id}")


def run_formal_deep(
    context: dict,
    *,
    model_id: str,
    fold_id: str,
    seed: int,
    device: str,
    run_dir: Path,
) -> None:
    protocol = context["protocol"]
    training = protocol["deep_training"]
    set_determinism(seed)
    data = load_fold(fold_id, project_root=context["root"], normalize=True, verify_hashes=True)
    dataset = FoldTrainPatchDataset(
        data,
        fault_frac=training["sampler"]["fold_fault_center_fraction"][fold_id],
        n=training["patches_per_epoch"],
        seed=seed,
        patch=protocol["data"]["patch_pixels"],
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    target_device = torch.device(device)
    model = build_deep_model(model_id).to(target_device)
    optimizer, scheduler = deep_optimizer_scheduler(model, protocol)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            training["loss"]["fold_positive_weight"][fold_id], device=target_device
        )
    )
    history = []
    best_ap = -np.inf
    best_epoch = 0
    wait = 0
    checkpoint_path = run_dir / "checkpoint.pt"
    for epoch in range(1, training["maximum_epochs"] + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        epoch_start = time.time()
        for x, y, mask in loader:
            x, y, mask = x.to(target_device), y.to(target_device), mask.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).squeeze(1)
            selected = mask > 0
            loss = criterion(logits[selected], y[selected].float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["optimizer"]["gradient_clip_norm"]
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        validation_probability = sliding_predict_fold(
            model,
            data,
            target_device,
            role="validation",
            patch=protocol["data"]["patch_pixels"],
            stride=protocol["data"]["inference_stride_pixels"],
            tta=False,
        )
        validation_metrics = evaluation_metrics(data["y"], validation_probability, data["val"])
        validation_ap = validation_metrics["average_precision"]
        improved = validation_ap > best_ap + training["checkpoint"]["minimum_delta"]
        if improved:
            best_ap = validation_ap
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "protocol_sha256": context["protocol_sha256"],
                    "model_id": model_id,
                    "fold_id": fold_id,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_ap": validation_ap,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                checkpoint_path,
            )
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / batches,
                "validation_average_precision": validation_ap,
                "validation_roc_auc": validation_metrics["roc_auc"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "best_epoch": best_epoch,
                "seconds": time.time() - epoch_start,
            }
        )
        scheduler.step()
        if (
            epoch >= training["minimum_epochs_before_early_stop"]
            and wait >= training["checkpoint"]["patience_epochs"]
        ):
            break
    write_csv(run_dir / "history.csv", history)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if checkpoint["protocol_sha256"] != context["protocol_sha256"]:
        raise RuntimeError("Best checkpoint protocol hash mismatch")
    model.load_state_dict(checkpoint["model_state"])
    validation_probability = sliding_predict_fold(
        model, data, target_device, role="validation", tta=False
    )
    validation_threshold, validation_f1 = best_threshold(
        data["y"], validation_probability, data["val"]
    )
    test_probability = sliding_predict_fold(model, data, target_device, role="test", tta=False)
    validation_metrics = evaluation_metrics(data["y"], validation_probability, data["val"])
    test_metrics = evaluation_metrics(data["y"], test_probability, data["test"])
    np.save(run_dir / "validation_probability.npy", validation_probability.astype(np.float32))
    np.save(run_dir / "test_probability.npy", test_probability.astype(np.float32))
    save_pr_curve(run_dir / "pr_curve.npz", data["y"], test_probability, data["test"])
    write_json(
        run_dir / "result.json",
        {
            "model_id": model_id,
            "model_kind": "deep",
            "fold_id": fold_id,
            "seed": seed,
            "protocol_sha256": context["protocol_sha256"],
            "trainable_parameters": trainable_parameters(model),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "validation": validation_metrics,
            "validation_selected_threshold_descriptive": validation_threshold,
            "validation_selected_f1_descriptive": validation_f1,
            "test": test_metrics,
            "test_threshold_tuned": False,
            "test_time_augmentation": False,
        },
    )


def run_formal_rf(
    context: dict,
    *,
    model_id: str,
    fold_id: str,
    seed: int,
    run_dir: Path,
) -> None:
    protocol = context["protocol"]
    data = load_fold(fold_id, project_root=context["root"], normalize=True, verify_hashes=True)
    feature_cube = data["X"] if model_id == "rf_direct" else rf_features(data)
    training_mask = data["train"]
    training_features = feature_cube[training_mask]
    training_labels = (data["y"][training_mask] == 1).astype(np.int8)
    config = protocol["random_forest_training"]
    classifier = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        class_weight=config["class_weight"],
        n_jobs=config["n_jobs"],
        random_state=seed,
    )
    start = time.time()
    classifier.fit(training_features, training_labels)
    elapsed = time.time() - start
    joblib.dump(classifier, run_dir / "model.joblib")
    validation_probability = np.zeros(data["y"].shape, dtype=np.float32)
    test_probability = np.zeros(data["y"].shape, dtype=np.float32)
    validation_probability[data["val"]] = classifier.predict_proba(
        feature_cube[data["val"]]
    )[:, 1]
    test_probability[data["test"]] = classifier.predict_proba(feature_cube[data["test"]])[:, 1]
    write_csv(
        run_dir / "history.csv",
        [{"fit_seconds": elapsed, "trees": config["n_estimators"], "training_pixels": len(training_labels)}],
    )
    np.save(run_dir / "validation_probability.npy", validation_probability)
    np.save(run_dir / "test_probability.npy", test_probability)
    save_pr_curve(run_dir / "pr_curve.npz", data["y"], test_probability, data["test"])
    validation_threshold, validation_f1 = best_threshold(
        data["y"], validation_probability, data["val"]
    )
    write_json(
        run_dir / "result.json",
        {
            "model_id": model_id,
            "model_kind": "random_forest",
            "fold_id": fold_id,
            "seed": seed,
            "protocol_sha256": context["protocol_sha256"],
            "training_pixels": int(len(training_labels)),
            "features": int(training_features.shape[1]),
            "validation": evaluation_metrics(data["y"], validation_probability, data["val"]),
            "validation_selected_threshold_descriptive": validation_threshold,
            "validation_selected_f1_descriptive": validation_f1,
            "test": evaluation_metrics(data["y"], test_probability, data["test"]),
            "test_threshold_tuned": False,
            "test_time_augmentation": False,
        },
    )


def execute_formal(args) -> None:
    context = load_protocol_context(args.project_root)
    if args.confirm_protocol_sha != context["protocol_sha256"]:
        raise ValueError(
            "Formal execution requires --confirm-protocol-sha equal to the locked protocol"
        )
    kind = validate_formal_cell(context, args.model_id, args.fold_id, args.seed)
    final_dir = formal_output_directory(
        context, args.model_id, args.fold_id, args.seed
    )
    attempt_dir = final_dir.with_name(final_dir.name + f".attempt_{args.attempt:03d}")
    if final_dir.exists() or attempt_dir.exists():
        raise FileExistsError(f"Refusing to overwrite formal/attempt output: {final_dir}")
    attempt_dir.mkdir(parents=True)
    write_json(
        attempt_dir / "config_snapshot.json",
        {
            "protocol": context["protocol"],
            "protocol_sha256": context["protocol_sha256"],
            "model_id": args.model_id,
            "fold_id": args.fold_id,
            "seed": args.seed,
            "attempt": args.attempt,
            "device": args.device if kind == "deep" else "cpu",
            "code_sha256": code_hashes(context["root"]),
        },
    )
    write_json(
        attempt_dir / "environment.json",
        environment_record(args.device if kind == "deep" else "cpu"),
    )
    try:
        if kind == "deep":
            run_formal_deep(
                context,
                model_id=args.model_id,
                fold_id=args.fold_id,
                seed=args.seed,
                device=args.device,
                run_dir=attempt_dir,
            )
        else:
            run_formal_rf(
                context,
                model_id=args.model_id,
                fold_id=args.fold_id,
                seed=args.seed,
                run_dir=attempt_dir,
            )
        finalize_manifest(
            attempt_dir,
            {
                "scope": "formal_v3_run",
                "protocol_sha256": context["protocol_sha256"],
                "model_id": args.model_id,
                "model_kind": kind,
                "fold_id": args.fold_id,
                "seed": args.seed,
                "attempt": args.attempt,
                "completed": True,
            },
        )
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt_dir, final_dir)
        print(json.dumps({"completed": True, "output_dir": str(final_dir)}, indent=2))
    except Exception as error:
        write_json(
            attempt_dir / "failure.json",
            {
                "completed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "protocol_sha256": context["protocol_sha256"],
                "model_id": args.model_id,
                "fold_id": args.fold_id,
                "seed": args.seed,
                "attempt": args.attempt,
            },
        )
        raise


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    argument_parser.add_argument("--model-id", required=True)
    argument_parser.add_argument("--fold-id", required=True)
    argument_parser.add_argument("--seed", type=int, required=True)
    argument_parser.add_argument("--device", default="cuda:0")
    argument_parser.add_argument("--attempt", type=int, default=1)
    argument_parser.add_argument("--confirm-protocol-sha", required=True)
    execute_formal(argument_parser.parse_args())
