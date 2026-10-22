#!/usr/bin/env python3
"""Protocol-bound runner for the random-pixel holdout diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from models_v3 import build_deep_model, trainable_parameters  # noqa: E402
from random_holdout_common_v1 import (  # noqa: E402
    RandomTrainPatchDataset,
    full_grid_predict,
    load_random_split,
    protocol_context,
    sha256,
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def optimizer_scheduler(model: nn.Module, training: dict):
    cfg = training["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    maximum, warmup = training["maximum_epochs"], training["schedule"]["warmup_epochs"]
    minimum_factor = training["schedule"]["minimum_learning_rate"] / cfg["learning_rate"]

    def multiplier(epoch_index: int) -> float:
        if epoch_index < warmup:
            return (epoch_index + 1) / warmup
        progress = min(max((epoch_index - warmup) / max(maximum - warmup - 1, 1), 0.0), 1.0)
        return minimum_factor + (1.0 - minimum_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    truth = (y[mask] == 1).astype(np.int8)
    score = np.clip(probability[mask].astype(np.float64), 0, 1)
    result = {"n": int(len(truth)), "positive": int(truth.sum()),
              "positive_rate": float(truth.mean()),
              "average_precision": float(average_precision_score(truth, score)),
              "roc_auc": float(roc_auc_score(truth, score))}
    for fraction in (0.01, 0.05):
        n_top = max(1, int(math.ceil(len(score) * fraction)))
        top = np.argpartition(score, -n_top)[-n_top:]
        tp_top = int(truth[top].sum())
        label = f"top_{int(fraction * 100)}pct"
        result[f"precision_{label}"] = tp_top / n_top
        result[f"recall_{label}"] = tp_top / max(int(truth.sum()), 1)
    predicted = score >= 0.5
    tp = int((predicted & (truth == 1)).sum())
    fp = int((predicted & (truth == 0)).sum())
    fn = int((~predicted & (truth == 1)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result.update({"threshold": 0.5, "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
                   "precision_at_0_5": precision, "recall_at_0_5": recall,
                   "tp_at_0_5": tp, "fp_at_0_5": fp, "fn_at_0_5": fn})
    return result


def best_threshold(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    truth, score = (y[mask] == 1), probability[mask]
    best = (0.5, -1.0)
    for threshold in np.linspace(0, 1, 51):
        predicted = score >= threshold
        tp, fp, fn = int((predicted & truth).sum()), int((predicted & ~truth).sum()), int((~predicted & truth).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best


def environment(device: str) -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": device}
    if device.startswith("cuda"):
        index = torch.device(device).index or 0
        result.update({"gpu_index": index, "gpu_name": torch.cuda.get_device_name(index),
                       "gpu_total_memory_bytes": torch.cuda.get_device_properties(index).total_memory})
    return result


def code_hashes(root: Path) -> dict:
    relatives = ["02_scripts/modeling/17_run_random_holdout_v1.py",
                 "02_scripts/modeling/random_holdout_common_v1.py",
                 "02_scripts/modeling/models_v3.py", "02_scripts/modeling/temporal_cmt.py",
                 "02_scripts/modeling/cross_modal.py", "03_configs/random_holdout_v1_protocol.json"]
    return {path: sha256(root / path) for path in relatives}


def finalize(run_dir: Path, identity: dict) -> None:
    hashes = {path.name: sha256(path) for path in sorted(run_dir.iterdir())
              if path.is_file() and path.name != "run_manifest.json"}
    write_json(run_dir / "run_manifest.json", {**identity, "file_sha256": hashes})


def run(args) -> None:
    root = args.project_root.resolve()
    context = protocol_context(root, args.confirm_protocol_sha)
    protocol, digest = context["protocol"], context["sha256"]
    split_ids = [x["split_id"] for x in protocol["splits"]]
    if args.model_id not in protocol["models"] or args.split_id not in split_ids or args.seed not in protocol["seeds"]:
        raise ValueError("Requested cell is outside the frozen 3 x 4 x 4 matrix")
    base = (root / "04_results/random_holdout_v1_smoke" / digest if args.smoke else
            root / "01_data/interim/results_random_holdout_v1" / digest)
    final_dir = base / args.model_id / args.split_id / f"seed_{args.seed}"
    attempt = final_dir.with_name(final_dir.name + ".attempt")
    if final_dir.exists() or attempt.exists():
        raise FileExistsError(f"Refusing to overwrite run output: {final_dir}")
    attempt.mkdir(parents=True)
    started = time.time()
    try:
        set_determinism(args.seed)
        data = load_random_split(root, args.split_id, digest)
        training = dict(protocol["training"])
        if args.smoke:
            training.update({"patches_per_epoch": 32, "maximum_epochs": 1,
                             "minimum_epochs_before_early_stop": 1})
        dataset = RandomTrainPatchDataset(data, n=training["patches_per_epoch"], seed=args.seed,
                                          patch=training["patch_pixels"])
        loader = torch.utils.data.DataLoader(dataset, batch_size=training["batch_size"], shuffle=False,
                                             num_workers=0, drop_last=True)
        device = torch.device(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(device)
        model = build_deep_model(args.model_id).to(device)
        optimizer, scheduler = optimizer_scheduler(model, training)
        positive = int((data["train"] & (data["y"] == 1)).sum())
        negative = int((data["train"] & (data["y"] == 0)).sum())
        pos_weight = negative / positive
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        history, best_ap, best_epoch, wait = [], -np.inf, 0, 0
        checkpoint_path = attempt / "checkpoint.pt"
        for epoch in range(1, training["maximum_epochs"] + 1):
            model.train()
            total_loss, batches, epoch_started = 0.0, 0, time.time()
            for x, y, mask in loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).squeeze(1)
                selected = mask > 0
                if not selected.any():
                    raise RuntimeError("Empty loss mask")
                loss = criterion(logits[selected], y[selected].float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["optimizer"]["gradient_clip_norm"])
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            probability = full_grid_predict(model, data["X"], device,
                                             patch=training["patch_pixels"],
                                             stride=protocol["inference"]["stride_pixels"],
                                             floor=protocol["inference"]["blend_floor"],
                                             batch_size=protocol["inference"]["window_batch_size"])
            validation = metrics(data["y"], probability, data["val"])
            improved = validation["average_precision"] > best_ap + training["checkpoint"]["minimum_delta"]
            if improved:
                best_ap, best_epoch, wait = validation["average_precision"], epoch, 0
                torch.save({"protocol_sha256": digest, "model_id": args.model_id, "split_id": args.split_id,
                            "seed": args.seed, "epoch": epoch, "validation_ap": best_ap,
                            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict()}, checkpoint_path)
            else:
                wait += 1
            history.append({"epoch": epoch, "training_loss": total_loss / batches,
                            "validation_average_precision": validation["average_precision"],
                            "validation_roc_auc": validation["roc_auc"],
                            "learning_rate": optimizer.param_groups[0]["lr"], "best_epoch": best_epoch,
                            "seconds": time.time() - epoch_started})
            scheduler.step()
            if not args.smoke and epoch >= training["minimum_epochs_before_early_stop"] and wait >= training["checkpoint"]["patience_epochs"]:
                break
        write_csv(attempt / "history.csv", history)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["protocol_sha256"] != digest:
            raise RuntimeError("Checkpoint protocol mismatch")
        model.load_state_dict(checkpoint["model_state"])
        final_probability = full_grid_predict(model, data["X"], device,
                                              patch=training["patch_pixels"],
                                              stride=protocol["inference"]["stride_pixels"],
                                              floor=protocol["inference"]["blend_floor"],
                                              batch_size=protocol["inference"]["window_batch_size"])
        np.save(attempt / "probability.npy", final_probability)
        threshold, f1 = best_threshold(data["y"], final_probability, data["val"])
        result = {"scope": "smoke_validation_only" if args.smoke else "formal_random_holdout_diagnostic_v1",
                  "protocol_sha256": digest, "model_id": args.model_id, "model_kind": "deep",
                  "split_id": args.split_id, "source_fold": data["source_fold"], "seed": args.seed,
                  "trainable_parameters": trainable_parameters(model), "epochs_completed": len(history),
                  "best_epoch": best_epoch, "training_positive_weight": pos_weight,
                  "training_center_positive_fraction": dataset.positive_fraction,
                  "validation": metrics(data["y"], final_probability, data["val"]),
                  "validation_selected_threshold_descriptive": threshold,
                  "validation_selected_f1_descriptive": f1,
                  "known_spatial_context_overlap": True,
                  "test_labels_accessed": not args.smoke, "elapsed_seconds": time.time() - started}
        if not args.smoke:
            result["test"] = metrics(data["y"], final_probability, data["test"])
            truth = (data["y"][data["test"]] == 1).astype(np.int8)
            precision, recall, thresholds = precision_recall_curve(truth, final_probability[data["test"]])
            np.savez_compressed(attempt / "pr_curve.npz", precision=precision, recall=recall, thresholds=thresholds)
        write_json(attempt / "result.json", result)
        write_json(attempt / "environment.json", environment(args.device))
        write_json(attempt / "config_snapshot.json", {"protocol": protocol, "protocol_sha256": digest,
                   "model_id": args.model_id, "split_id": args.split_id, "seed": args.seed,
                   "device": args.device, "smoke": args.smoke, "code_sha256": code_hashes(root)})
        finalize(attempt, {"status": "complete", "scope": result["scope"], "protocol_sha256": digest,
                 "model_id": args.model_id, "split_id": args.split_id, "seed": args.seed})
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        attempt.rename(final_dir)
        print(json.dumps({"status": "complete", "output": str(final_dir),
                          "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
    except Exception as error:
        write_json(attempt / "failure.json", {"status": "failed", "type": type(error).__name__,
                   "error": str(error), "protocol_sha256": digest, "model_id": args.model_id,
                   "split_id": args.split_id, "seed": args.seed})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
