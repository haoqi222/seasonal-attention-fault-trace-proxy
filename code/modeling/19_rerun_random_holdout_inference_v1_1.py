#!/usr/bin/env python3
"""Re-infer one random-holdout cell with the exact V4.2 blend floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from models_v3 import build_deep_model  # noqa: E402
from random_holdout_common_v1 import full_grid_predict, load_random_split  # noqa: E402


PROTOCOL = Path("03_configs/random_holdout_v1_1_inference_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict:
    truth = (y[mask] == 1).astype(np.int8)
    score = np.asarray(probability[mask], dtype=np.float64)
    result = {"n": int(len(truth)), "positive": int(truth.sum()), "positive_rate": float(truth.mean()),
              "average_precision": float(average_precision_score(truth, score)),
              "roc_auc": float(roc_auc_score(truth, score))}
    for fraction in (0.01, 0.05):
        count = max(1, int(math.ceil(len(score) * fraction)))
        top = np.argpartition(score, -count)[-count:]
        tp = int(truth[top].sum()); label = f"top_{int(fraction * 100)}pct"
        result[f"precision_{label}"] = tp / count
        result[f"recall_{label}"] = tp / max(int(truth.sum()), 1)
    predicted = score >= 0.5
    tp = int((predicted & (truth == 1)).sum()); fp = int((predicted & (truth == 0)).sum())
    fn = int((~predicted & (truth == 1)).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    result.update({"threshold": 0.5, "f1_at_0_5": 2 * precision * recall / max(precision + recall, 1e-9),
                   "precision_at_0_5": precision, "recall_at_0_5": recall,
                   "tp_at_0_5": tp, "fp_at_0_5": fp, "fn_at_0_5": fn})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--model-id", required=True); parser.add_argument("--split-id", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(); protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); digest = sha256(protocol_path)
    if protocol["status"] != "frozen_before_v1_1_inference" or args.confirm_protocol_sha != digest:
        raise ValueError("V1.1 protocol confirmation failed")
    if args.model_id not in protocol["models"] or args.split_id not in protocol["splits"] or args.seed not in protocol["seeds"]:
        raise ValueError("Cell outside frozen matrix")
    parent_sha = protocol["parent_random_training_protocol_sha256"]
    parent = root / "01_data/interim/results_random_holdout_v1" / parent_sha / args.model_id / args.split_id / f"seed_{args.seed}"
    parent_manifest = json.loads((parent / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = parent / "checkpoint.pt"
    if parent_manifest.get("status") != "complete" or parent_manifest.get("protocol_sha256") != parent_sha:
        raise RuntimeError("Parent random-holdout run is not accepted")
    if sha256(checkpoint_path) != parent_manifest["file_sha256"]["checkpoint.pt"]:
        raise RuntimeError("Parent checkpoint hash mismatch")
    destination = root / "01_data/interim/results_random_holdout_v1_1" / digest / args.model_id / args.split_id / f"seed_{args.seed}"
    attempt = destination.with_name(destination.name + ".attempt")
    if destination.exists() or attempt.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    attempt.mkdir(parents=True)
    data = load_random_split(root, args.split_id, parent_sha)
    device = torch.device(args.device); torch.cuda.set_device(device)
    model = build_deep_model(args.model_id).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if any(checkpoint.get(key) != value for key, value in {"protocol_sha256": parent_sha,
           "model_id": args.model_id, "split_id": args.split_id, "seed": args.seed}.items()):
        raise RuntimeError("Parent checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state"])
    probability = full_grid_predict(model, data["X"], device,
                                    patch=protocol["change_scope"]["window_size_pixels"],
                                    stride=protocol["change_scope"]["stride_pixels"],
                                    floor=protocol["blend"]["floor"],
                                    batch_size=protocol["change_scope"]["window_batch_size"])
    np.save(attempt / "probability.npy", probability)
    truth = (data["y"][data["test"]] == 1).astype(np.int8)
    precision, recall, thresholds = precision_recall_curve(truth, probability[data["test"]])
    np.savez_compressed(attempt / "pr_curve.npz", precision=precision, recall=recall, thresholds=thresholds)
    result = {"scope": "formal_random_holdout_v1_1_inference", "protocol_sha256": digest,
              "parent_training_protocol_sha256": parent_sha, "parent_checkpoint_sha256": sha256(checkpoint_path),
              "model_id": args.model_id, "split_id": args.split_id, "source_fold": data["source_fold"],
              "seed": args.seed, "blend_floor": protocol["blend"]["floor"],
              "validation": metrics(data["y"], probability, data["val"]),
              "test": metrics(data["y"], probability, data["test"])}
    write_json(attempt / "result.json", result)
    write_json(attempt / "config_snapshot.json", protocol)
    env = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
           "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "device": args.device, "gpu_name": torch.cuda.get_device_name(device)}
    write_json(attempt / "environment.json", env)
    hashes = {p.name: sha256(p) for p in sorted(attempt.iterdir()) if p.is_file()}
    write_json(attempt / "run_manifest.json", {"status": "complete", "scope": result["scope"],
               "protocol_sha256": digest, "model_id": args.model_id, "split_id": args.split_id,
               "seed": args.seed, "file_sha256": hashes})
    destination.parent.mkdir(parents=True, exist_ok=True); attempt.rename(destination)
    print(json.dumps({"status": "complete", "output": str(destination)}))


if __name__ == "__main__":
    main()
