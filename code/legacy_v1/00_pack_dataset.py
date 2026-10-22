#!/usr/bin/env python
"""
阶段5：打包模型数据集（空间分块 + 特征堆叠 + 掩膜 + 归一化统计）。

空间分块（D11/D12 后的设计确认，2026-08-28）：
  Z1 训练-北  y[0:500]          (0–50 km)
      缓冲1   y[500:550]        (5 km，丢弃，切断空间自相关泄漏)
  Z2 训练-中  y[550:1200]       (55–120 km，Llano 核心)
      缓冲2   y[1200:1250]      (5 km，丢弃)
  Z3 测试-南  y[1250:1840]      (125–184 km，Socorro 南段 + 最南负样本富集区)

特征：4 季 × 9 波段，按“波段外层、季节内层”排为 36 通道：
  0:4   ST 春/夏/秋/冬               (热模态)
  4:8   red 春/夏/秋/冬
  8:12  nir08 ...
  12:16 swir16 ...
  16:20 ndvi ...
  20:24 ndbi ...
  24:28 ndwi ...
  28:32 bsi ...
  count 仅用于有效掩膜，不进入预测特征，避免模型学习轨道覆盖/数据可得性伪影。

DEM 不进模型（用户确认），留作事后归因分析。

输出：01_data/interim/dataset/
  X.npy            (1840, 539, 32) float32，无效像元(count==0)填 0
  y.npy            (1840, 539) uint8，1=正(≤200m) 2=忽略(200-1000m) 0=背景
  valid.npy        (1840, 539) bool，四季 count>0
  zone.npy         (1840, 539) uint8，1=TRAIN 2=VAL 3=TEST 0=丢弃(缓冲/无效)
  norm_stats.json  训练区(Z1+Z2)每通道 mean/std（z-score 统计，防泄漏）
  meta.json        波段/分区/网格说明

用法：
  .venv/bin/python 02_scripts/modeling/00_pack_dataset.py
"""
import json
import os

import numpy as np
import rioxarray as rx

COMP_DIR = "01_data/interim/gee_comp_v2"
LABEL_PATH = "01_data/interim/label_inferred_fault_v2.tif"
OUT_DIR = "01_data/interim/dataset_v2"

SEASONS = ["spring", "summer", "autumn", "winter"]
# GEE 合成脚本的波段顺序（02_gee_seasonal_composite.py band_list）
BANDS_IN = ["ST", "red", "nir08", "swir16", "ndvi", "ndbi", "ndwi", "bsi", "count"]
FEATURE_BANDS = ["ST", "red", "nir08", "swir16", "ndvi", "ndbi", "ndwi", "bsi"]
# 空间分块（行范围，北=0 南=1839）
ZONES = {
    "Z1_train_north": (0, 500),
    "buffer1":        (500, 550),
    "Z2_train_mid":   (550, 1200),
    "buffer2":        (1200, 1250),
    "Z3_test_south":  (1250, 1840),
}
TRAIN_ZONES = ("Z1_train_north", "Z2_train_mid")
TRAIN_RANGES = ((0, 500), (550, 1050))
VAL_RANGE = (1050, 1200)
TEST_RANGE = (1250, 1840)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读 4 季栅格，按波段→季节堆叠
    seasonal = {}
    for s in SEASONS:
        da = rx.open_rasterio(os.path.join(COMP_DIR, f"gee_{s}_full.tif"))
        seasonal[s] = da.values.astype(np.float32)  # (9, ny, nx)

    ny, nx = seasonal[SEASONS[0]].shape[1:]
    X = np.zeros((ny, nx, len(FEATURE_BANDS) * len(SEASONS)), dtype=np.float32)
    for bi, band in enumerate(FEATURE_BANDS):
        for si, s in enumerate(SEASONS):
            X[..., bi * len(SEASONS) + si] = seasonal[s][bi]
    print(f"特征张量: {X.shape} ({len(FEATURE_BANDS)} 科学波段 × {len(SEASONS)} 季；count不入模)")

    # 2. 有效掩膜 = 四季 count>0
    valid = np.ones((ny, nx), dtype=bool)
    for si, s in enumerate(SEASONS):
        count_band = seasonal[s][BANDS_IN.index("count")]
        valid &= count_band > 0
    X[~valid] = 0.0
    print(f"有效像元: {valid.sum():,}/{ny*nx:,} ({valid.mean()*100:.1f}%)")

    # 3. 标签 + 分区
    y = rx.open_rasterio(LABEL_PATH).values[0].astype(np.uint8)
    assert y.shape == (ny, nx), f"标签网格 {y.shape} 与特征 {X.shape[:2]} 不一致"

    zone = np.zeros((ny, nx), dtype=np.uint8)
    for r0, r1 in TRAIN_RANGES:
        zone[r0:r1] = 1
    zone[VAL_RANGE[0]:VAL_RANGE[1]] = 2
    zone[TEST_RANGE[0]:TEST_RANGE[1]] = 3
    zone[~valid] = 0  # 无效像元并入丢弃

    # 4. 归一化统计：只用训练区(Z1+Z2)有效像元
    train_mask = np.zeros((ny, nx), dtype=bool)
    for r0, r1 in TRAIN_RANGES:
        train_mask[r0:r1] = True
    train_mask &= valid
    stats = {"mean": [], "std": []}
    for c in range(X.shape[2]):
        vals = X[..., c][train_mask]
        stats["mean"].append(float(vals.mean()))
        stats["std"].append(float(vals.std()))
    print("\n每通道统计（训练区）：")
    for c in range(X.shape[2]):
        b = FEATURE_BANDS[c // len(SEASONS)]
        s = SEASONS[c % len(SEASONS)]
        print(f"  [{c:2d}] {b:6s} {s:6s}  mean={stats['mean'][c]:9.3f}  std={stats['std'][c]:8.3f}")

    # 5. 写盘
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "valid.npy"), valid)
    np.save(os.path.join(OUT_DIR, "zone.npy"), zone)
    with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    meta = {
        "grid": {"ny": ny, "nx": nx, "dtype": "float32"},
        "crs": "EPSG:32613", "pixel_m": 100.0,
        "bands": FEATURE_BANDS, "seasons": SEASONS,
        "channel_order": "band-major, season-minor (0:4=ST sp/su/au/wi ... 28:32=BSI)",
        "count_role": "validity mask only; excluded from model features",
        "splits": {"train": TRAIN_RANGES, "val": VAL_RANGE, "test": TEST_RANGE,
                   "buffers": ((500, 550), (1200, 1250))},
        "valid_mask": "count>0 in all 4 seasons",
        "fill_invalid": 0.0,
        "normalization": "z-score with train-zone stats in norm_stats.json, applied at load",
        "dem_in_model": False,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 6. 汇总
    print(f"\n分区汇总（行 {ny} 总）:")
    names = {1: "TRAIN", 2: "VAL", 3: "TEST", 0: "丢弃(缓冲/无效)"}
    for zi in (1, 2, 3, 0):
        m = zone == zi
        pos = int((y[m] == 1).sum()); ig = int((y[m] == 2).sum()); bg = int((y[m] == 0).sum())
        print(f"  {names[zi]:14s}: 像元={m.sum():7,d}  正={pos:6,d} ({pos/m.sum()*100 if m.sum() else 0:4.1f}%)  "
              f"忽略={ig:6,d} 背景={bg:7,d}")
    print(f"\n数据包已保存 → {OUT_DIR}/")


if __name__ == "__main__":
    main()
