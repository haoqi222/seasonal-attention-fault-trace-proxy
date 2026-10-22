#!/usr/bin/env python
"""
阶段4：把 GEE 按片下载的季节合成拼接成整研宄区栅格。

输入：01_data/interim/gee_comp_v2/gee_{season}_tile{i}.tif（EPSG:32613，100m，相邻纬度片）
输出：01_data/interim/gee_comp_v2/gee_{season}_full.tif

用法：
  .venv/bin/python 02_scripts/gee/04_mosaic_seasons.py [season...]
"""
import os
import sys

import rioxarray as rx
import xarray as xr

from rioxarray.merge import merge_arrays

COMP_DIR = "01_data/interim/gee_comp_v2"
SEASONS = ["spring", "summer", "autumn", "winter"]


def main():
    seasons = sys.argv[1:] or SEASONS
    for sname in seasons:
        tiles = sorted(
            f for f in os.listdir(COMP_DIR)
            if f.startswith(f"gee_{sname}_tile") and f.endswith(".tif"))
        if not tiles:
            print(f"[{sname}] 无片，跳过")
            continue
        print(f"[{sname}] 拼接 {len(tiles)} 片 ...")
        das = [rx.open_rasterio(os.path.join(COMP_DIR, t)) for t in tiles]  # 保留全部波段
        merged = merge_arrays(das)
        out = os.path.join(COMP_DIR, f"gee_{sname}_full.tif")
        merged.rio.to_raster(out)
        shape = merged.shape
        print(f"  ✅ {sname} 全图：{shape}，已保存 {out}")
        # 校验：输出是否完整覆盖研究区（无大空洞）
        import numpy as np
        v = merged.values
        print(f"     波段数={merged.shape[0]}，有效值占比：{np.isfinite(v).mean()*100:.1f}%")
        print("     保留分片文件，待第二版数据与实验验收后再决定是否清理。")


if __name__ == "__main__":
    main()
