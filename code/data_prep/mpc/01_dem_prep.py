#!/usr/bin/env python
"""
阶段3/4 通用：构建研究区 Copernicus DEM GLO-30 + 坡度。

流程：
  1. 从 MPC `cop-dem-glo-30` 搜索覆盖研究区 bbox 的所有 1°×1° 瓦片。
  2. 每瓦片按 bbox 做服务端窗口裁剪（EPSG:4326）。
  3. 合并为马赛克。
  4. 重投影到 EPSG:32613（UTM 13N），目标分辨率 100 m。
  5. 派生坡度（坡向可选）。
  6. 保存研究区 DEM（GeoTIFF）。

研究区：新墨西哥 Rio Grande Rift 中央裂谷带（D7）。
  bbox = [-108.0, 33.0, -106.0, 36.5]（经度, 纬度）

用法：
  .venv/bin/python 02_scripts/mpc/01_dem_prep.py
"""
import json
import os
import urllib.request

import numpy as np
import rioxarray as rx
import xarray as xr
from pyproj import Transformer

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION = "cop-dem-glo-30"

# 研究区（经度, 纬度）
BBOX_LONLAT = [-108.0, 33.0, -106.0, 36.5]
TARGET_CRS = "EPSG:32613"
TARGET_RES = 100.0  # m
OUT_DIR = "01_data/interim"


def token(account, container):
    req = urllib.request.Request(f"{SAS}/{account}/{container}")
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read().decode())["token"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"== 搜索 {COLLECTION} 瓦片（bbox={BBOX_LONLAT}）==")
    payload = {"collections": [COLLECTION], "bbox": BBOX_LONLAT, "limit": 500}
    req = urllib.request.Request(STAC, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    feats = data.get("features", [])
    print(f"覆盖瓦片数：{len(feats)}")
    for f in feats:
        print("  -", f["id"])

    if not feats:
        print("❌ 无瓦片覆盖，检查 bbox")
        return

    # 逐瓦片窗口裁剪（EPSG:4326，直接传经纬度 bbox）
    crops = []
    for f in feats:
        id_ = f["id"]
        # 跳过只擦边、退化（与 bbox 交集可忽略）的瓦片，避免单列/单行裁剪
        tb = f["bbox"]  # [minx,miny,maxx,maxy]
        ix_min = max(tb[0], BBOX_LONLAT[0]); ix_max = min(tb[2], BBOX_LONLAT[2])
        iy_min = max(tb[1], BBOX_LONLAT[1]); iy_max = min(tb[3], BBOX_LONLAT[3])
        if (ix_max - ix_min) < 0.01 or (iy_max - iy_min) < 0.01:
            print(f"  跳过（仅擦边/无有效覆盖）：{id_}")
            continue
        href = f["assets"]["data"]["href"]
        netloc = href.split("blob.core.windows.net/")[0].split("//")[1]
        account = netloc.split(".")[0]
        container = href.split("blob.core.windows.net/")[1].split("/")[0]
        signed = href + "?" + token(account, container)
        print(f"  读取 {id_} ...")
        da = rx.open_rasterio(signed)
        cr = da.rio.clip_box(minx=ix_min, miny=iy_min, maxx=ix_max, maxy=iy_max)
        # 过滤退化裁剪
        if cr.shape[1] < 2 or cr.shape[2] < 2:
            print(f"  跳过（退化 {cr.shape}）：{id_}")
            da.close()
            continue
        crops.append(cr)
        da.close()
    if not crops:
        print("❌ 无有效瓦片裁剪")
        return
    print(f"有效裁剪瓦片数：{len(crops)}")

    # 合并马赛克：先在地理坐标(EPSG:4326)下 merge（瓦片相邻/不相交），再一次性重投影
    print("合并马赛克（EPSG:4326 merge）...")
    from rioxarray.merge import merge_arrays
    single = [c.isel(band=0) for c in crops]
    dem_geo = merge_arrays(single)
    print(f"  地理合并：{dem_geo.shape}，bounds={dem_geo.rio.bounds()}")

    # 一次性重投影到 UTM13N @ 100m
    print(f"重投影到 {TARGET_CRS} @ {TARGET_RES:.0f} m ...")
    dem_utm = dem_geo.rio.reproject(TARGET_CRS, resolution=TARGET_RES, resampling="bilinear")
    dem_utm = dem_utm.rename("dem")

    dem_path = os.path.join(OUT_DIR, "study_dem_utm13n_100m.tif")
    dem_utm.rio.to_raster(dem_path)
    arr = dem_utm.values
    v = arr[np.isfinite(arr)]
    print(f"✅ 研究区 DEM：{arr.shape}，高程 {v.min():.0f}~{v.max():.0f} m（均值 {v.mean():.0f}）")
    print(f"   已保存：{dem_path}")

    # 坡度
    print("计算坡度 ...")
    z = dem_utm.values.astype(float)
    gy, gx = np.gradient(z, TARGET_RES, TARGET_RES)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    slope_da = dem_utm.copy(data=slope_deg)
    slope_da = slope_da.rename("slope")
    slope_path = os.path.join(OUT_DIR, "study_slope_utm13n_100m.tif")
    slope_da.rio.to_raster(slope_path)
    print(f"✅ 研究区坡度：{slope_deg.shape}，已保存：{slope_path}")


if __name__ == "__main__":
    main()
