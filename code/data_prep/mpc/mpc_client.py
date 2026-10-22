#!/usr/bin/env python
"""
MPC（Microsoft Planetary Computer）数据访问客户端——项目统一复用模块。

封装三件事（D9 实测验证）：
  1. get_token(account, container)   —— 拿容器级 SAS 签名 token（HTTP GET，匿名，免登录）
  2. sign_url(blob_url, account, container) —— 给 blob URL 追加签名
  3. read_crop(...)                  —— 用 rioxarray 按 bbox 做服务端 windowed COG 裁剪

用途：下载 Landsat Collection 2 Level-2（地表温度 + 光学）到本地 GeoTIFF，供四季合成与建模。

波段映射：MPC `landsat-c2-l2` 集合（Source: USGS Landsat C2 L2，经 MPC 分发）。
  lwir11   -> 地表温度 ST_B10（K，uint16，scale=0.00341802, offset=149），30m 网格（100m 热红外）
  red/blue/green/nir08/swir16/swir22/coastal -> SR 光学

用法：
  from mpc_client import sign_url, read_crop, get_token
  url = sign_url(blob_url, "landsateuwest", "landsat-c2")
  da  = read_crop(url, bbox_lonlat=[minx,miny,maxx,maxy], out_crs="EPSG:32613")
"""
import os

# == 代理配置（D10）：所有出网流量走本地 Clash 7897，urllib 与 GDAL(/vsicurl/) 均认 ==
# 可被环境变量 MPC_PROXY 覆盖；设为 "" 可禁用代理。
_MPC_PROXY = os.environ.get("MPC_PROXY", "http://127.0.0.1:7897")
if _MPC_PROXY:
    for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.setdefault(_k, _MPC_PROXY)
    _GEO_PROXY = os.environ.get("GDAL_HTTP_PROXY", "YES")
    if _GEO_PROXY:
        os.environ.setdefault("GDAL_HTTP_PROXY", _MPC_PROXY)

import json
import time
import urllib.request

import numpy as np
import rioxarray as rx
from pyproj import Transformer

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_BASE = "https://planetarycomputer.microsoft.com/api/sas/v1/token"

_TOKEN_CACHE: dict = {}


# ---------------------------------------------------------------- token
def get_token(account: str, container: str, retry: int = 4) -> str:
    """取容器级 SAS token（带简单缓存）。匿名，无 subscription key。"""
    url = f"{SAS_BASE}/{account}/{container}"
    cached = _TOKEN_CACHE.get(url)
    if cached and time.time() < cached["exp"] - 60:
        return cached["tok"]
    req = urllib.request.Request(url)
    last = None
    for i in range(retry):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read().decode())
            _TOKEN_CACHE[url] = {"tok": d["token"], "exp": 1e18}  # token 有效期~1h，每次现取更稳
            return d["token"]
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"get_token 失败 {account}/{container}: {last}")


def sign_url(blob_url: str, account: str, container: str) -> str:
    """给 Azure blob URL 追加 SAS token。若已带 st/se/sp 参数则视为已签名。"""
    if any(k in blob_url for k in ("st=", "se=", "sp=")):
        return blob_url
    return blob_url + "?" + get_token(account, container)


# ---------------------------------------------------------------- search
def search_scenes(collections, bbox_lonlat, start, end, max_results=100000):
    """STAC 搜索，返回场景 dict 列表（分页自动处理）。"""
    payload = {
        "collections": [collections],
        "bbox": list(bbox_lonlat),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 500,
    }
    out_rows, total = [], 0
    while True:
        req = urllib.request.Request(STAC_SEARCH, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        for f in data.get("features", []):
            p = f["properties"]
            out_rows.append({
                "item_id": f.get("id", ""),
                "acq": p.get("datetime", "")[:10],
                "cloud": p.get("eo:cloud_cover"),
                "path": p.get("landsat:wrs_path", ""),
                "row": p.get("landsat:wrs_row", ""),
                "platform": p.get("platform", ""),
                "href_ST_B10": f["assets"].get("lwir11", {}).get("href", ""),
                "href_red": f["assets"].get("red", {}).get("href", ""),
                "href_nir": f["assets"].get("nir08", {}).get("href", ""),
                "href_swir16": f["assets"].get("swir16", {}).get("href", ""),
            })
        total += len(f.get("features", []))
        nxt = next((l for l in data.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or nxt is None or total >= max_results:
            break
        # MPC next 的 body 带新 token
        if "body" in nxt:
            payload = nxt["body"]
        else:
            break
    return out_rows


# ---------------------------------------------------------------- crop read
def _latlon_to_utm13n(bbox_lonlat):
    """经纬度 bbox -> UTM 13N 米坐标 (xmin, ymin, xmax, ymax)，用于 EPSG:32613 栅格裁剪。"""
    minx, miny, maxx, maxy = bbox_lonlat
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
    xmin, ymax = tr.transform(minx, maxy)
    xmax, ymin = tr.transform(maxx, miny)
    return xmin, ymin, xmax, ymax


def read_crop(blob_url: str, bbox_lonlat, bands=("lwir11",), account="landsateuwest",
              container="landsat-c2") -> dict:
    """
    从 MPC blob 按 bbox 做服务端窗口裁剪，返回 {band: DataArray}。

    - blob_url 须为 STAC item 的 asset"裸" URL（未签名，函数内部签名）。
    - bbox_lonlat = [minx, miny, maxx, maxy]，经纬度。
    - 输出 KEEP 原始单位（ST 为 K，光学为反射率 scaled），由调用方换算。
    """
    signed = sign_url(blob_url, account, container)
    xmin, ymin, xmax, ymax = _latlon_to_utm13n(bbox_lonlat)
    out = {}
    for b in bands:
        da = rx.open_rasterio(signed)
        crop = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
        crop.rio.write_crs("EPSG:32613", inplace=True)
        out[b] = crop
    return out


def st_to_kelvin(arr: np.ndarray, scale=0.00341802, offset=149.0) -> np.ndarray:
    """ST_B10 uint16 -> Kelvin float，0 视为 nodata -> nan。"""
    out = np.where(arr > 0, arr * scale + offset, np.nan)
    return out.astype(np.float64)


def save_crop(da, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    da.rio.to_raster(path)


# ---------------------------------------------------------------- DEM
DEM_COLLECTION = "cop-dem-glo-30"


def _search_tiles(bbox_lonlat, collection=DEM_COLLECTION):
    """STAC 搜索覆盖 bbox 的 DEM 瓦片，返回 feature 列表（含 assets.data.href）。"""
    payload = {"collections": [collection], "bbox": list(bbox_lonlat), "limit": 500}
    req = urllib.request.Request(STAC_SEARCH, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return data.get("features", [])


def dem_for_bbox(bbox_lonlat, out_crs="EPSG:32613", out_res=100.0) -> "rx.DataArray":
    """
    搜索并合并覆盖 bbox 的 Copernicus DEM GLO-30，裁剪、重投影到 out_crs@out_res。
    返回 DataArray（高程，m）。适合小范围（如单条断层段）快速取 DEM。
    若 bbox 很小跨单个瓦片则极快。
    """
    feats = _search_tiles(bbox_lonlat)
    if not feats:
        raise RuntimeError("无 DEM 瓦片覆盖该 bbox")
    crops = []
    for f in feats:
        tb = f["bbox"]
        ix = (max(tb[0], bbox_lonlat[0]), min(tb[2], bbox_lonlat[2]),
              max(tb[1], bbox_lonlat[1]), min(tb[3], bbox_lonlat[3]))
        if (ix[1] - ix[0]) < 0.01 or (ix[3] - ix[2]) < 0.01:
            continue  # 擦边/退化瓦片跳过
        href = f["assets"]["data"]["href"]
        host = href.split("blob.core.windows.net/")[0].split("//")[1]
        account = host.split(".")[0]
        container = href.split("blob.core.windows.net/")[1].split("/")[0]
        signed = sign_url(href, account, container)
        da = rx.open_rasterio(signed)
        cr = da.rio.clip_box(minx=ix[0], miny=ix[2], maxx=ix[1], maxy=ix[3])
        da.close()
        if cr.shape[1] < 2 or cr.shape[2] < 2:
            continue
        crops.append(cr.isel(band=0))
    if not crops:
        raise RuntimeError("裁剪后无有效 DEM 瓦片")
    if len(crops) == 1:
        dem = crops[0]
    else:
        from rioxarray.merge import merge_arrays
        dem = merge_arrays(crops)
    dem = dem.rio.reproject(out_crs, resolution=out_res, resampling="bilinear")
    dem = dem.rename("dem")
    return dem


# ---------------------------------------------------------------- test
if __name__ == "__main__":
    print("== 自检：签名 + 窗口裁剪 ST_B10 ==")
    blob = ("https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/oli-tirs/"
            "2021/034/036/LC08_L2SP_034036_20210211_20210302_02_T1/"
            "LC08_L2SP_034036_20210211_20210302_02_T1_ST_B10.TIF")
    bbox = [-106.95, 34.50, -106.70, 35.25]
    d = read_crop(blob, bbox, bands=("lwir11",))
    arr = d["lwir11"].values[0]
    k = st_to_kelvin(arr)
    v = k[np.isfinite(k)]
    print("裁剪形状:", d["lwir11"].shape, "| 有效值 %.1f%%" % (100 * np.mean(np.isfinite(k))))
    print("K: min=%.1f max=%.1f mean=%.1f (%.1f~%.1f °C)" %
          (v.min(), v.max(), v.mean(), v.min() - 273.15, v.max() - 273.15))
    save_crop(d["lwir11"], "01_data/raw/landsat_test/ST_crop_lwir11.tif")
    print("✅ 自检通过")
