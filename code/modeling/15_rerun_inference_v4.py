#!/usr/bin/env python3
"""V4 evaluation-only inference from the accepted V3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common_v3 import ROLE_CODES, inference_window_starts, load_fold  # noqa: E402
from models_v3 import DEEP_MODEL_IDS, build_deep_model, trainable_parameters  # noqa: E402


DEFAULT_PROTOCOL_REL = Path("03_configs/inference_v4_protocol.json")
V3_RESULT_ROOT = Path("01_data/interim/results_v3")
V4_RESULT_ROOT = Path("01_data/interim/results_v4")
PREFLIGHT_ROOT = Path("04_results/inference_v4_preflight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    result.update({
        "threshold": 0.5,
        "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "tp_at_0_5": tp,
        "fp_at_0_5": fp,
        "fn_at_0_5": fn,
    })
    return result


def best_threshold(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.asarray(probability[selected], dtype=np.float64)
    best = (0.5, -1.0)
    for threshold in np.linspace(0, 1, 51):
        predicted = score >= threshold
        tp = int((predicted & (truth == 1)).sum())
        fp = int((predicted & (truth == 0)).sum())
        fn = int((~predicted & (truth == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best[1]:
            best = (float(threshold), float(f1))
    return best


def blend_weight(patch: int, floor: float) -> np.ndarray:
    coordinate = np.arange(patch, dtype=np.float64)
    axis = floor + (1.0 - floor) * np.sin(np.pi * (coordinate + 0.5) / patch) ** 2
    weight = np.outer(axis, axis)
    if not np.all(np.isfinite(weight)) or np.min(weight) <= 0 or np.max(weight) > 1:
        raise RuntimeError("Invalid V4 blend weight")
    return weight


def weighted_predict(model, data: dict, device: torch.device, *, role: str,
                     patch: int, stride: int, floor: float) -> np.ndarray:
    windows = inference_window_starts(data, role=role, patch=patch, stride=stride)
    weight = blend_weight(patch, floor)
    accumulator = np.zeros(data["partition"].shape, dtype=np.float64)
    denominator = np.zeros(data["partition"].shape, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for y0, x0 in windows:
            patch_array = data["X"][y0:y0 + patch, x0:x0 + patch]
            tensor = torch.from_numpy(patch_array.transpose(2, 0, 1).copy())[None].to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            accumulator[y0:y0 + patch, x0:x0 + patch] += probability * weight
            denominator[y0:y0 + patch, x0:x0 + patch] += weight
    target = data["partition"] == ROLE_CODES[role]
    if not np.all(denominator[target] > 0) or np.any(denominator[~target] > 0):
        raise RuntimeError(f"Invalid weighted inference coverage for {role}")
    output = np.divide(accumulator, denominator, out=np.zeros_like(accumulator), where=denominator > 0)
    if not np.all(np.isfinite(output)) or float(output.min()) < 0 or float(output.max()) > 1:
        raise RuntimeError("Weighted probability outside [0,1] or non-finite")
    return output.astype(np.float32)


def save_pr(path: Path, y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> None:
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    precision, recall, thresholds = precision_recall_curve(truth, probability[selected])
    np.savez_compressed(path, precision=precision, recall=recall, thresholds=thresholds)


def protocol_context(root: Path, protocol_file: Path, confirm_sha: str) -> dict:
    path = protocol_file if protocol_file.is_absolute() else root / protocol_file
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256(path)
    if protocol["status"] != "frozen_before_v4_inference":
        raise ValueError("V4 protocol is not frozen")
    if confirm_sha != digest:
        raise ValueError(f"Protocol confirmation mismatch: {confirm_sha} != {digest}")
    return {"path": path, "protocol": protocol, "sha256": digest}


def environment(device: str) -> dict:
    value = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "device": device,
    }
    if device.startswith("cuda"):
        index = torch.device(device).index or 0
        value["gpu_name"] = torch.cuda.get_device_name(index)
    return value


def run(args) -> None:
    root = args.project_root.resolve()
    context = protocol_context(root, args.protocol_file, args.confirm_protocol_sha)
    protocol = context["protocol"]
    if args.model_id not in DEEP_MODEL_IDS:
        raise ValueError(f"V4 rerun accepts deep models only: {args.model_id}")
    if args.fold_id not in ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]:
        raise ValueError(args.fold_id)
    if args.seed not in (0, 1, 2, 3):
        raise ValueError(args.seed)
    if args.preflight and (args.seed != 0 or args.role != "validation"):
        raise ValueError("Preflight is validation-only and seed 0")
    if not args.preflight and args.role != "both":
        raise ValueError("Formal V4 execution must generate validation and test together")

    parent_sha = protocol["parent_v3"]["training_protocol_sha256"]
    parent_dir = root / V3_RESULT_ROOT / parent_sha / args.model_id / args.fold_id / f"seed_{args.seed}"
    checkpoint_path = parent_dir / "checkpoint.pt"
    parent_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if not parent_manifest["completed"] or parent_manifest["protocol_sha256"] != parent_sha:
        raise ValueError("Parent V3 run is not accepted complete")
    if sha256(checkpoint_path) != parent_manifest["file_sha256"]["checkpoint.pt"]:
        raise ValueError("Parent checkpoint hash mismatch")

    destination = ((root / PREFLIGHT_ROOT / context["sha256"]) if args.preflight else
                   (root / V4_RESULT_ROOT / context["sha256"]))
    destination = destination / args.model_id / args.fold_id / f"seed_{args.seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    attempt = destination.with_name(destination.name + ".attempt")
    if attempt.exists():
        raise FileExistsError(f"Attempt already exists: {attempt}")
    attempt.mkdir(parents=True)
    started = time.time()
    set_determinism(args.seed)
    data = load_fold(args.fold_id, project_root=root, normalize=True, verify_hashes=True)
    device = torch.device(args.device)
    model = build_deep_model(args.model_id).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["protocol_sha256"] != parent_sha or checkpoint["model_id"] != args.model_id:
        raise ValueError("Parent checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state"])
    patch = int(protocol["change_scope"]["window_size_pixels"])
    stride = int(protocol["change_scope"]["stride_pixels"])
    floor = float(protocol["blend"]["floor"])
    roles = ["validation"] if args.role == "validation" else ["validation", "test"]
    result = {
        "model_id": args.model_id, "model_kind": "deep", "fold_id": args.fold_id,
        "seed": args.seed, "v4_protocol_sha256": context["sha256"],
        "parent_v3_protocol_sha256": parent_sha,
        "parent_checkpoint_path": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
        "parent_checkpoint_sha256": sha256(checkpoint_path),
        "trainable_parameters": trainable_parameters(model),
        "blend": protocol["blend"], "preflight": args.preflight,
    }
    output_hashes = {}
    for role in roles:
        probability = weighted_predict(model, data, device, role=role, patch=patch, stride=stride, floor=floor)
        filename = f"{role}_probability.npy"
        np.save(attempt / filename, probability)
        output_hashes[filename] = sha256(attempt / filename)
        role_mask = data["val"] if role == "validation" else data["test"]
        result[role] = metrics(data["y"], probability, role_mask)
        if role == "test":
            save_pr(attempt / "pr_curve.npz", data["y"], probability, role_mask)
            output_hashes["pr_curve.npz"] = sha256(attempt / "pr_curve.npz")
    threshold, f1 = best_threshold(data["y"], np.load(attempt / "validation_probability.npy"), data["val"])
    result["validation_selected_threshold_descriptive"] = threshold
    result["validation_selected_f1_descriptive"] = f1
    result["elapsed_seconds"] = time.time() - started
    write_json(attempt / "result.json", result)
    write_json(attempt / "environment.json", environment(args.device))
    write_json(attempt / "config_snapshot.json", protocol)
    for name in ("result.json", "environment.json", "config_snapshot.json"):
        output_hashes[name] = sha256(attempt / name)
    write_json(attempt / "run_manifest.json", {
        "status": "complete", "v4_protocol_sha256": context["sha256"],
        "model_id": args.model_id, "fold_id": args.fold_id, "seed": args.seed,
        "preflight": args.preflight, "roles": roles, "files": output_hashes,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt.rename(destination)
    print(json.dumps({"status": "complete", "output": str(destination), "seconds": result["elapsed_seconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL_REL)
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--role", choices=["validation", "both"], required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
