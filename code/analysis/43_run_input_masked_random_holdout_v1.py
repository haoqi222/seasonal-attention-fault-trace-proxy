#!/usr/bin/env python3
"""Run the frozen input-level masking diagnostic on accepted random-holdout cells."""

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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from models_v3 import build_deep_model  # noqa: E402
from random_holdout_common_v1 import _starts, blend_weight, load_random_split  # noqa: E402


PROTOCOL = Path("03_configs/random_holdout_input_mask_v1_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
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


def masked_full_grid_predict(model, X: np.ndarray, input_mask: np.ndarray, device: torch.device,
                             *, patch: int, stride: int, floor: float, batch_size: int) -> np.ndarray:
    windows = [(y0, x0) for y0 in _starts(X.shape[0], patch, stride)
               for x0 in _starts(X.shape[1], patch, stride)]
    weight = blend_weight(patch, floor)
    accumulator = np.zeros(X.shape[:2], dtype=np.float64)
    denominator = np.zeros(X.shape[:2], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            group = windows[offset:offset + batch_size]
            arrays = []
            for y0, x0 in group:
                block = X[y0:y0 + patch, x0:x0 + patch].copy()
                block[input_mask[y0:y0 + patch, x0:x0 + patch]] = 0.0
                arrays.append(block.transpose(2, 0, 1))
            probabilities = torch.sigmoid(model(torch.from_numpy(np.stack(arrays).astype(np.float32)).to(device)))[:, 0].cpu().numpy()
            for (y0, x0), probability in zip(group, probabilities):
                accumulator[y0:y0 + patch, x0:x0 + patch] += probability * weight
                denominator[y0:y0 + patch, x0:x0 + patch] += weight
    if not np.all(denominator > 0):
        raise RuntimeError("Masked inference left uncovered pixels")
    output = accumulator / denominator
    if not np.all(np.isfinite(output)) or output.min() < 0 or output.max() > 1:
        raise RuntimeError("Invalid masked probability")
    return output.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--model-id", required=True); parser.add_argument("--split-id", required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(); protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); digest = sha256(protocol_path)
    if protocol["status"] != "frozen_before_inference" or args.confirm_protocol_sha != digest:
        raise ValueError("Input-mask protocol confirmation failed")
    if args.model_id not in protocol["models"] or args.split_id not in protocol["splits"] or args.seed not in protocol["seeds"]:
        raise ValueError("Cell outside frozen matrix")
    parent_sha = protocol["parent_inference_protocol_sha256"]
    parent = root / "01_data/interim/results_random_holdout_v1" / "6391a2680c2173f0deda638f2b8c317d1d681def25b73e7da9df9ccf92148627" / args.model_id / args.split_id / f"seed_{args.seed}"
    checkpoint_path = parent / "checkpoint.pt"
    parent_result = root / "01_data/interim/results_random_holdout_v1_1" / parent_sha / args.model_id / args.split_id / f"seed_{args.seed}" / "result.json"
    if not checkpoint_path.is_file() or not parent_result.is_file():
        raise FileNotFoundError("Accepted checkpoint or unmasked V1.1 result is missing")
    destination = root / "04_results/random_holdout_input_mask_v1" / digest / args.model_id / args.split_id / f"seed_{args.seed}"
    attempt = destination.with_name(destination.name + ".attempt")
    if destination.exists() or attempt.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    attempt.mkdir(parents=True)
    data = load_random_split(root, args.split_id, "6391a2680c2173f0deda638f2b8c317d1d681def25b73e7da9df9ccf92148627")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = build_deep_model(args.model_id).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    probability = masked_full_grid_predict(model, data["X"], data["test"], device,
                                           patch=128, stride=64, floor=0.01, batch_size=16)
    masked_metrics = metrics(data["y"], probability, data["test"])
    baseline = json.loads(parent_result.read_text(encoding="utf-8"))
    result = {"scope": "formal_random_holdout_v1_input_level_masking", "protocol_sha256": digest,
              "parent_inference_protocol_sha256": parent_sha, "model_id": args.model_id, "split_id": args.split_id,
              "source_fold": data["source_fold"], "seed": args.seed, "masked_test": masked_metrics,
              "unmasked_test": baseline["test"],
              "delta_masked_minus_unmasked": {key: masked_metrics[key] - baseline["test"][key]
                                              for key in ("average_precision", "roc_auc", "f1_at_0_5")}}
    np.save(attempt / "masked_probability.npy", probability)
    write_json(attempt / "result.json", result); write_json(attempt / "config_snapshot.json", protocol)
    write_json(attempt / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version(), "device": args.device,
               "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})
    hashes = {p.name: sha256(p) for p in sorted(attempt.iterdir()) if p.is_file()}
    write_json(attempt / "run_manifest.json", {"status": "complete", "scope": result["scope"],
               "protocol_sha256": digest, "model_id": args.model_id, "split_id": args.split_id,
               "seed": args.seed, "file_sha256": hashes})
    destination.parent.mkdir(parents=True, exist_ok=True); attempt.rename(destination)
    print(json.dumps({"status": "complete", "output": str(destination), "average_precision": masked_metrics["average_precision"]}))


if __name__ == "__main__": main()
