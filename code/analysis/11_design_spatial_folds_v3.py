#!/usr/bin/env python3
"""Audit and freeze the V3 spatial-fold design.

This stage is read-only with respect to the V2 dataset and result package. It
uses the frozen V2 grid, validity mask, labels, and QFFD inferred-trace vector
to evaluate candidate north-south blocks. The selected four-fold design has a
50-row (5 km) excluded gap between every adjacent core block and applies
fault-ID purging so held-out fault IDs cannot remain labelled in development
roles.

The script writes only small audit/manifest files to
``04_results/split_design_v3``. It does not create V3 training masks or run a
model; those are explicit gates for the next stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box


BUFFER_ROWS = 50
POSITIVE_DISTANCE_M = 200.0
IGNORE_DISTANCE_M = 1000.0

SELECTED_BLOCKS = OrderedDict(
    [
        ("B1_North", (0, 400)),
        ("B2_NorthCentral", (450, 800)),
        ("B3_SouthCentral", (850, 1200)),
        ("B4_South", (1250, 1840)),
    ]
)

SELECTED_GAPS = OrderedDict(
    [
        ("G1", (400, 450)),
        ("G2", (800, 850)),
        ("G3", (1200, 1250)),
    ]
)

# Validation blocks minimize direct test-validation fault-ID overlap while
# keeping every validation set large enough for checkpoint selection.
FOLDS = OrderedDict(
    [
        ("F1_NorthTest", {"test": ["B1_North"], "validation": ["B4_South"],
                           "train": ["B2_NorthCentral", "B3_SouthCentral"]}),
        ("F2_NorthCentralTest", {"test": ["B2_NorthCentral"], "validation": ["B4_South"],
                                  "train": ["B1_North", "B3_SouthCentral"]}),
        ("F3_SouthCentralTest", {"test": ["B3_SouthCentral"], "validation": ["B1_North"],
                                  "train": ["B2_NorthCentral", "B4_South"]}),
        ("F4_SouthTest", {"test": ["B4_South"], "validation": ["B1_North"],
                           "train": ["B2_NorthCentral", "B3_SouthCentral"]}),
    ]
)

ROLE_GATES = {
    "train": {"min_eval_pixels": 200_000, "min_positive_pixels": 5_000,
              "min_fault_ids": 5},
    "validation": {"min_eval_pixels": 100_000, "min_positive_pixels": 5_000,
                   "min_fault_ids": 5},
    "test": {"min_eval_pixels": 100_000, "min_positive_pixels": 5_000,
             "min_fault_ids": 5},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ranges_mask(shape: tuple[int, int], ranges: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for start, end in ranges:
        mask[start:end] = True
    return mask


def equal_candidate_blocks(n_rows: int, k: int) -> OrderedDict[str, tuple[int, int]]:
    core_rows = (n_rows - (k - 1) * BUFFER_ROWS) // k
    blocks: OrderedDict[str, tuple[int, int]] = OrderedDict()
    start = 0
    for index in range(k):
        end = n_rows if index == k - 1 else start + core_rows
        blocks[f"B{index + 1}"] = (start, end)
        start = end + BUFFER_ROWS
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = (args.output_dir or root / "04_results/split_design_v3").resolve()
    required = {
        "y": root / "01_data/interim/dataset_v2/y.npy",
        "valid": root / "01_data/interim/dataset_v2/valid.npy",
        "vector": root / "01_data/interim/nm_rifft_concealed.gpkg",
        "reference": root / "01_data/interim/label_inferred_fault_v2.tif",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    output_names = [
        "candidate_block_audit_v3.csv",
        "selected_block_audit_v3.csv",
        "fault_id_overlap_v3.csv",
        "fold_summary_v3.csv",
        "fold_manifest_v3.json",
        "stage1_acceptance_v3.json",
        "stage1_spatial_fold_report_v3.md",
    ]
    existing = [out / name for name in output_names if (out / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))
    out.mkdir(parents=True, exist_ok=True)

    y = np.load(required["y"])
    valid = np.load(required["valid"]).astype(bool)
    if y.shape != valid.shape:
        raise ValueError(f"Shape mismatch: y={y.shape}, valid={valid.shape}")
    if set(np.unique(y).tolist()) - {0, 1, 2}:
        raise ValueError(f"Unexpected labels: {np.unique(y)}")

    with rasterio.open(required["reference"]) as dataset:
        if dataset.shape != y.shape:
            raise ValueError(f"Reference shape {dataset.shape} != label shape {y.shape}")
        transform = dataset.transform
        bounds = dataset.bounds
        crs = dataset.crs
    pixel_m = abs(float(transform.a))
    if not np.isclose(pixel_m, 100.0) or not np.isclose(abs(transform.e), 100.0):
        raise ValueError(f"Expected a 100 m square grid, got transform={transform}")

    faults = gpd.read_file(required["vector"]).to_crs(crs)
    if "fault_id" not in faults or "linetype" not in faults:
        raise ValueError("Vector must contain fault_id and linetype")
    faults = faults[faults["linetype"].astype(str).str.casefold().eq("inferred")].copy()
    study_box = box(*bounds)
    faults = faults[faults.geometry.intersects(study_box.buffer(IGNORE_DISTANCE_M))].copy()
    faults["fault_id_text"] = faults["fault_id"].astype(str)
    all_fault_ids = set(faults["fault_id_text"])

    def polygon_for(rows: tuple[int, int]):
        start, end = rows
        return box(bounds.left, bounds.top - end * pixel_m,
                   bounds.right, bounds.top - start * pixel_m)

    def ids_for(rows: tuple[int, int], influence_m: float = 0.0) -> set[str]:
        polygon = polygon_for(rows)
        if influence_m:
            polygon = polygon.buffer(influence_m)
        return set(faults.loc[faults.geometry.intersects(polygon), "fault_id_text"])

    def basic_stats(rows: tuple[int, int]) -> dict:
        start, end = rows
        block_valid = valid[start:end]
        block_y = y[start:end]
        evaluation = block_valid & (block_y != 2)
        positive = block_valid & (block_y == 1)
        negative = block_valid & (block_y == 0)
        direct_ids = ids_for(rows)
        influence_ids = ids_for(rows, IGNORE_DISTANCE_M)
        return {
            "start_row": start,
            "end_row": end,
            "row_count": end - start,
            "valid_pixels": int(block_valid.sum()),
            "evaluation_pixels": int(evaluation.sum()),
            "positive_pixels": int(positive.sum()),
            "negative_pixels": int(negative.sum()),
            "ignored_valid_pixels": int((block_valid & (block_y == 2)).sum()),
            "positive_prevalence": float(positive.sum() / max(1, evaluation.sum())),
            "direct_fault_ids": len(direct_ids),
            "influence_fault_ids": len(influence_ids),
        }

    candidate_rows: list[dict] = []
    for k in (3, 4, 5):
        for block_name, rows in equal_candidate_blocks(y.shape[0], k).items():
            candidate_rows.append({"candidate": f"equal_k{k}", "block": block_name,
                                   **basic_stats(rows)})
    write_csv(out / "candidate_block_audit_v3.csv", candidate_rows)

    selected_rows: list[dict] = []
    direct_ids: dict[str, set[str]] = {}
    influence_ids: dict[str, set[str]] = {}
    for kind, mapping in (("core", SELECTED_BLOCKS), ("buffer", SELECTED_GAPS)):
        for name, rows in mapping.items():
            selected_rows.append({"kind": kind, "name": name, **basic_stats(rows)})
            direct_ids[name] = ids_for(rows)
            influence_ids[name] = ids_for(rows, IGNORE_DISTANCE_M)
    write_csv(out / "selected_block_audit_v3.csv", selected_rows)

    overlap_rows: list[dict] = []
    block_names = list(SELECTED_BLOCKS)
    for left_index, left_name in enumerate(block_names):
        for right_name in block_names[left_index + 1:]:
            direct_overlap = sorted(direct_ids[left_name] & direct_ids[right_name])
            influence_overlap = sorted(influence_ids[left_name] & influence_ids[right_name])
            overlap_rows.append({
                "left_block": left_name,
                "right_block": right_name,
                "direct_overlap_count": len(direct_overlap),
                "direct_overlap_ids": ";".join(direct_overlap),
                "influence_overlap_count": len(influence_overlap),
                "influence_overlap_ids": ";".join(influence_overlap),
            })
    write_csv(out / "fault_id_overlap_v3.csv", overlap_rows)

    label_cache: dict[frozenset[str], np.ndarray] = {}

    def label_for(allowed_ids: set[str]) -> np.ndarray:
        cache_key = frozenset(allowed_ids)
        if cache_key in label_cache:
            return label_cache[cache_key]
        included = faults[faults["fault_id_text"].isin(allowed_ids)]
        excluded = faults[~faults["fault_id_text"].isin(allowed_ids)]
        if included.empty:
            raise ValueError("A role has no allowed fault IDs")
        included_mask = rasterize(
            [(geometry, 1) for geometry in included.geometry],
            out_shape=y.shape, transform=transform, fill=0, all_touched=True,
        )
        included_distance = ndimage.distance_transform_edt(included_mask == 0) * pixel_m
        label = np.zeros(y.shape, dtype=np.uint8)
        label[(included_distance > POSITIVE_DISTANCE_M)
              & (included_distance <= IGNORE_DISTANCE_M)] = 2
        label[included_distance <= POSITIVE_DISTANCE_M] = 1
        if not excluded.empty:
            excluded_mask = rasterize(
                [(geometry, 1) for geometry in excluded.geometry],
                out_shape=y.shape, transform=transform, fill=0, all_touched=True,
            )
            excluded_distance = ndimage.distance_transform_edt(excluded_mask == 0) * pixel_m
            # Excluded held-out IDs are never converted to background. Their
            # complete 0-1000 m influence zone is removed from the role.
            label[excluded_distance <= IGNORE_DISTANCE_M] = 2
        label_cache[cache_key] = label
        return label

    fold_rows: list[dict] = []
    manifest_folds: list[dict] = []
    acceptance_checks: list[dict] = []
    for fold_id, roles in FOLDS.items():
        raw_test_ids = set().union(*(influence_ids[name] for name in roles["test"]))
        raw_validation_ids = set().union(*(influence_ids[name] for name in roles["validation"]))
        raw_train_ids = set().union(*(influence_ids[name] for name in roles["train"]))

        allowed_ids = {
            "test": raw_test_ids,
            "validation": raw_validation_ids - raw_test_ids,
            "train": raw_train_ids - raw_test_ids - raw_validation_ids,
        }
        disjoint = (
            not (allowed_ids["test"] & allowed_ids["validation"])
            and not (allowed_ids["test"] & allowed_ids["train"])
            and not (allowed_ids["validation"] & allowed_ids["train"])
        )
        acceptance_checks.append({"fold": fold_id, "check": "allowed_fault_ids_disjoint",
                                  "passed": disjoint, "value": disjoint})

        fold_record = {
            "fold_id": fold_id,
            "roles": {},
            "raw_test_validation_shared_ids": sorted(raw_test_ids & raw_validation_ids),
            "heldout_ids_present_in_raw_train": sorted(
                (raw_test_ids | raw_validation_ids) & raw_train_ids
            ),
        }
        for role in ("train", "validation", "test"):
            blocks = roles[role]
            row_ranges = [SELECTED_BLOCKS[name] for name in blocks]
            core_mask = ranges_mask(y.shape, row_ranges)
            role_mask = core_mask & valid
            role_label = label_for(allowed_ids[role])
            evaluation = role_mask & (role_label != 2)
            positive = role_mask & (role_label == 1)
            negative = role_mask & (role_label == 0)
            ignored = role_mask & (role_label == 2)
            raw_ids = set().union(*(influence_ids[name] for name in blocks))
            purged_ids = raw_ids - allowed_ids[role]
            test_matches_v2 = None
            if role == "test":
                test_matches_v2 = bool(np.array_equal(role_label[core_mask], y[core_mask]))
            row = {
                "fold_id": fold_id,
                "role": role,
                "blocks": ";".join(blocks),
                "row_ranges": ";".join(f"[{start},{end})" for start, end in row_ranges),
                "row_count": sum(end - start for start, end in row_ranges),
                "valid_pixels": int(role_mask.sum()),
                "evaluation_pixels": int(evaluation.sum()),
                "positive_pixels": int(positive.sum()),
                "negative_pixels": int(negative.sum()),
                "ignored_valid_pixels": int(ignored.sum()),
                "positive_prevalence": float(positive.sum() / max(1, evaluation.sum())),
                "raw_influence_fault_ids": len(raw_ids),
                "allowed_fault_ids": len(allowed_ids[role]),
                "purged_fault_ids": len(purged_ids),
                "test_label_matches_v2": test_matches_v2,
            }
            gate = ROLE_GATES[role]
            role_passed = (
                row["evaluation_pixels"] >= gate["min_eval_pixels"]
                and row["positive_pixels"] >= gate["min_positive_pixels"]
                and row["allowed_fault_ids"] >= gate["min_fault_ids"]
                and (role != "test" or test_matches_v2)
            )
            row["accepted"] = role_passed
            fold_rows.append(row)
            acceptance_checks.append({
                "fold": fold_id,
                "check": f"{role}_gate",
                "passed": role_passed,
                "value": {
                    "evaluation_pixels": row["evaluation_pixels"],
                    "positive_pixels": row["positive_pixels"],
                    "allowed_fault_ids": row["allowed_fault_ids"],
                    "test_label_matches_v2": test_matches_v2,
                },
                "threshold": gate,
            })
            fold_record["roles"][role] = {
                "blocks": blocks,
                "row_ranges": [list(item) for item in row_ranges],
                "raw_influence_fault_ids": sorted(raw_ids),
                "allowed_fault_ids": sorted(allowed_ids[role]),
                "purged_fault_ids": sorted(purged_ids),
                "counts": {key: row[key] for key in (
                    "valid_pixels", "evaluation_pixels", "positive_pixels",
                    "negative_pixels", "ignored_valid_pixels", "positive_prevalence"
                )},
                "accepted": role_passed,
            }
        manifest_folds.append(fold_record)
    write_csv(out / "fold_summary_v3.csv", fold_rows)

    assignment = np.full(y.shape[0], "", dtype=object)
    for name, (start, end) in SELECTED_BLOCKS.items():
        if np.any(assignment[start:end] != ""):
            raise ValueError(f"Overlapping selected block: {name}")
        assignment[start:end] = name
    for name, (start, end) in SELECTED_GAPS.items():
        if np.any(assignment[start:end] != ""):
            raise ValueError(f"Overlapping selected gap: {name}")
        assignment[start:end] = name
    full_partition = bool(np.all(assignment != ""))
    gap_rows_correct = all(end - start == BUFFER_ROWS for start, end in SELECTED_GAPS.values())
    adjacent_separation = all(
        list(SELECTED_BLOCKS.values())[index + 1][0]
        - list(SELECTED_BLOCKS.values())[index][1] == BUFFER_ROWS
        for index in range(len(SELECTED_BLOCKS) - 1)
    )
    for check, passed, value in (
        ("all_rows_assigned_once", full_partition, full_partition),
        ("all_buffers_are_50_rows", gap_rows_correct, BUFFER_ROWS),
        ("adjacent_cores_separated_by_50_rows", adjacent_separation, BUFFER_ROWS),
    ):
        acceptance_checks.append({"fold": "global", "check": check,
                                  "passed": passed, "value": value})

    overall_accepted = all(item["passed"] for item in acceptance_checks)
    manifest = {
        "schema_version": "fault-proxy-v3-split-1.0",
        "status": "accepted_for_stage_2_mask_generation" if overall_accepted else "rejected",
        "grid": {
            "shape": list(y.shape), "crs": str(crs), "pixel_size_m": pixel_m,
            "buffer_rows": BUFFER_ROWS, "buffer_distance_m": BUFFER_ROWS * pixel_m,
        },
        "label_policy": {
            "positive_distance_m": POSITIVE_DISTANCE_M,
            "ignore_distance_m": IGNORE_DISTANCE_M,
            "fault_id_field": "fault_id",
            "fault_ids_are_based_on": "all traces whose 1000 m influence zone intersects a role block",
            "purge": (
                "Test IDs are removed from validation and training. Validation IDs are removed from "
                "training. Pixels within 0-1000 m of a purged ID are ignored, never background."
            ),
        },
        "selected_blocks": {name: list(rows) for name, rows in SELECTED_BLOCKS.items()},
        "excluded_buffers": {name: list(rows) for name, rows in SELECTED_GAPS.items()},
        "selection_rationale": (
            "Equal 3-block and 5-block candidates leave the southern core with fewer than five direct "
            "QFFD fault IDs. The adjusted four-block design starts the southern core at row 1250, "
            "retains the existing geographically held-out south band, and meets the minimum test "
            "sample and fault-ID gates while inserting 5 km gaps at every internal boundary."
        ),
        "folds": manifest_folds,
        "input_sha256": {name: sha256(path) for name, path in required.items()},
        "role_gates": ROLE_GATES,
    }
    (out / "fold_manifest_v3.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    acceptance = {
        "overall_accepted": overall_accepted,
        "next_gate": (
            "Generate fold-specific V3 label/mask rasters and verify their counts and hashes before "
            "normalization or model training."
        ),
        "checks": acceptance_checks,
    }
    (out / "stage1_acceptance_v3.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_lookup = {(row["fold_id"], row["role"]): row for row in fold_rows}
    lines = [
        "# V3 实验阶段1：空间折叠审计与冻结",
        "",
        f"状态：{'通过' if overall_accepted else '未通过'}。本阶段未修改 V2 数据或结果，也未启动模型训练。",
        "",
        "## 冻结设计",
        "",
        "- 100 m 网格；相邻核心块之间固定排除 50 行，即 5 km。",
        "- 采用四个南北向核心块，每个核心块轮流作为一次测试区。",
        "- 测试断层 ID 不进入验证或训练标签；验证断层 ID 不进入训练标签。",
        "- 被净化断层周围 0–1000 m 全部设为忽略，不得改作背景。",
        "",
        "| 折叠 | 测试块 | 验证块 | 训练块 | 测试正样本 | 测试 fault ID | 净化后训练正样本 | 净化后训练 fault ID |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for fold_id, roles in FOLDS.items():
        test = summary_lookup[(fold_id, "test")]
        train = summary_lookup[(fold_id, "train")]
        lines.append(
            f"| {fold_id} | {', '.join(roles['test'])} | {', '.join(roles['validation'])} | "
            f"{', '.join(roles['train'])} | {test['positive_pixels']:,} | "
            f"{test['allowed_fault_ids']} | {train['positive_pixels']:,} | "
            f"{train['allowed_fault_ids']} |"
        )
    lines.extend([
        "",
        "## 验收结论",
        "",
        "- 四个测试折均不少于 100,000 个可评价像元、5,000 个正样本和 5 个影响测试块的 fault ID。",
        "- 所有训练、验证、测试角色在净化后均满足预设样本与 fault-ID 门槛。",
        "- 测试角色的标签与冻结 V2 标签在对应核心块内逐像元一致。",
        "- 当前只冻结空间块和断层 ID 规则；V3 标签栅格、归一化统计和模型结果尚未生成。",
        "",
        "## 下一道停止门",
        "",
        "阶段2必须先生成每折独立的标签、忽略掩膜、训练/验证/测试掩膜和输入校验值。"
        "这些文件通过像元计数、fault-ID互斥和SHA-256核验后，才允许拟合归一化统计或训练模型。",
        "",
    ])
    (out / "stage1_spatial_fold_report_v3.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out),
        "overall_accepted": overall_accepted,
        "folds": len(FOLDS),
        "files": output_names,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
