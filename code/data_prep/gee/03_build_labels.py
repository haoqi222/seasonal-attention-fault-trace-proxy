#!/usr/bin/env python
"""
阶段4：生成“推断位置第四纪断层”代理标签，网格对齐第二版 GEE 合成栅格。

标签定义（在 100 m 网格上）：
  1 = 正样本（距隐伏断层 ≤ 200 m，断层核心带）
  2 = 忽略区（200 m < 距断层 ≤ 1000 m，位置不确定环，从损失中排除）
  0 = 背景/负样本（距断层 > 1000 m）

输入：
  - USGS QFFD 推断位置断层：01_data/interim/nm_rifft_concealed.gpkg
    注意：linetype=Inferred 表示位置由稀疏数据推断/外推，不等同于地质意义的隐伏或埋藏断层。
  - 参考栅格：01_data/interim/gee_comp_v2/gee_{season}_full.tif（任选一个作为网格模板）
输出：
  - 01_data/interim/label_inferred_fault_v2.tif   （1/2/0）
  - 01_data/interim/dist_to_inferred_fault_v2.tif （距断层距离，m）
  - 01_data/interim/inferred_fault_mask_v2.tif    （断层栅格掩膜）

用法：
  .venv/bin/python 02_scripts/gee/03_build_labels.py
  （参考栅格取自 gee_comp 中第一个 *_full.tif；缺失则用研究区 bbox 自建网格）
"""
import os
import sys

import geopandas as gpd
import numpy as np
import rioxarray as rx
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy import ndimage

COMP_DIR = "01_data/interim/gee_comp_v2"
CONCEALED = "01_data/interim/nm_rifft_concealed.gpkg"
OUT_DIR = "01_data/interim"
TARGET_CRS = "EPSG:32613"
BBOX = (-107.10, 33.70, -106.55, 35.35)  # (minLon,minLat,maxLon,maxLat)


def get_ref_grid():
    """找 gee_comp 中一个 *_full.tif 作网格模板；没有则按 bbox@100m 自建。"""
    for f in sorted(os.listdir(COMP_DIR)):
        if f.endswith("_full.tif"):
            da = rx.open_rasterio(os.path.join(COMP_DIR, f))
            return da
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xmin, ymax = tr.transform(BBOX[0], BBOX[3]); xmax, ymin = tr.transform(BBOX[2], BBOX[1])
    nx = int(round((xmax - xmin) / 100)); ny = int(round((ymax - ymin) / 100))
    da = rx.open_rasterio(os.path.join(COMP_DIR, os.listdir(COMP_DIR)[0])) if os.listdir(COMP_DIR) else None
    import xarray as xr, numpy as np
    dummy = xr.DataArray(np.zeros((1, ny, nx)), dims=("band", "y", "x"))
    dummy.rio.write_crs(TARGET_CRS, inplace=True)
    dummy.rio.write_transform(Affine(100, 0, xmin, 0, -100, ymax), inplace=True)
    return dummy


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = get_ref_grid()
    transform = ref.rio.transform()
    ny, nx = ref.shape[1], ref.shape[2]
    print(f"标签网格：{ny}x{nx}，transform={transform}")

    # 载入隐伏断层，裁剪到研究区
    f = gpd.read_file(CONCEALED).to_crs(TARGET_CRS)
    if len(f) == 0:
        print("无隐伏断层，退出"); sys.exit(1)
    from shapely.geometry import box
    b = box(*(transform.c, transform.f + ny * transform.e,
              transform.c + nx * transform.a, transform.f))
    f = f[f.geometry.intersects(b)]
    print(f"研究区内推断位置断层段数：{len(f)}")

    mask = rasterize([(g.geometry, 1) for g in f.itertuples()],
                     out_shape=(ny, nx), transform=transform, fill=0, all_touched=True)
    dist = ndimage.distance_transform_edt(mask == 0) * abs(transform.a)  # 米

    label = np.zeros((ny, nx), dtype=np.uint8)
    label[(dist > 200) & (dist <= 1000)] = 2   # 忽略区
    label[dist <= 200] = 1                       # 正样本
    # 背景(>1000m) 保持 0

    def save(arr, name, dtype=None, nodata=None):
        da = ref.isel(band=[0]).copy(data=arr[None].astype(dtype or arr.dtype))
        if nodata is not None:
            da.rio.write_nodata(nodata, inplace=True)
        da.rio.to_raster(os.path.join(OUT_DIR, name))

    save(label, "label_inferred_fault_v2.tif")
    save(dist, "dist_to_inferred_fault_v2.tif", dtype=np.float32)
    save(mask, "inferred_fault_mask_v2.tif")

    pos = (label == 1).sum(); ig = (label == 2).sum(); bg = (label == 0).sum()
    tot = ny * nx
    print(f"\n标签统计（总 {tot} 像元）：")
    print(f"  正样本(≤200m): {pos} ({pos/tot*100:.1f}%)")
    print(f"  忽略区(200-1000m): {ig} ({ig/tot*100:.1f}%)")
    print(f"  背景(>1000m): {bg} ({bg/tot*100:.1f}%)")


if __name__ == "__main__":
    main()
