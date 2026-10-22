#!/usr/bin/env python
"""
阶段4：用 GEE 云端合成全研究区 2014-2025 四季 LST + 光学 + 指数。

研究区：Rio Grande Rift 盆地内部埋藏断层（中段 Llano + 南段 Socorro），D11 定为正向信号区。
输出：每季一个 GeoTIFF（100 m，UTM 13N，EPSG:32613），含波段：
    ST (LST, K) / blue / green / red / nir08 / swir16 / swir22
    ndvi / ndbi / ndwi / bsi
    count (合成用有效观测数)

数据：Landsat 8/9 C2 L2（"LANDSAT/LC08/C02/T1_L2" + "LANDSAT/LC09/C02/T1_L2"）。
用 getDownloadURL 下载（区域在 100m 下足够小），经本地代理 7897。

前置：GEE 已认证（project=szh20260828），本地代理 7897 已开。
用法：
  .venv/bin/python 02_scripts/gee/02_gee_seasonal_composite.py [season...]
  （省略参数则跑全部 4 季）
"""
import os
import sys
import time

os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

import ee

PROJECT = "szh20260828"
OUT_DIR = "01_data/interim/gee_comp_v2"
# 研究区（中段 + 南段盆地联合 bbox）；D11 聚焦盆地内部埋藏断层
BBOX = (-107.10, 33.70, -106.55, 35.35)   # (minLon, minLat, maxLon, maxLat)
START, END = "2014-01-01", "2025-12-31"
SEASONS = {
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
    "winter": (12, 1, 2),
}
TARGET_CRS = "EPSG:32613"
SCALE = 100.0  # m
YEARS = list(range(2014, 2026))

# 波段定义
SR_BANDS = {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4",
            "nir08": "SR_B5", "swir16": "SR_B6", "swir22": "SR_B7"}
ST_BAND = "ST_B10"
OUTPUT_BANDS = ["ST", "red", "nir08", "swir16", "ndvi", "ndbi", "ndwi", "bsi"]


def mask_clouds(image):
    """C2 L2 QA：剔除填充值、膨胀云、卷云、云、云影、雪和饱和像元。"""
    qa = image.select("QA_PIXEL")
    # QA_PIXEL bits 0-5: fill, dilated cloud, cirrus, cloud, cloud shadow, snow.
    good = qa.bitwiseAnd(0b111111).eq(0)
    good = good.And(image.select("QA_RADSAT").eq(0))
    return image.updateMask(good)


def to_kelvin(image):
    return image.select(ST_BAND).multiply(0.00341802).add(149.0).rename("ST")


def add_indices(img):
    b4, b5, b6, b2 = "red", "nir08", "swir16", "blue"
    ndvi = img.expression("(b5-b4)/(b5+b4)", {"b5": img.select(b5), "b4": img.select(b4)}).rename("ndvi")
    ndbi = img.expression("(b6-b5)/(b6+b5)", {"b6": img.select(b6), "b5": img.select(b5)}).rename("ndbi")
    ndwi = img.expression("(b3-b5)/(b3+b5)", {"b3": img.select("green"), "b5": img.select(b5)}).rename("ndwi")
    bsi = img.expression("((b6+b4)-(b5+b2))/((b6+b4)+(b5+b2))",
                         {"b6": img.select(b6), "b4": img.select(b4),
                          "b5": img.select(b5), "b2": img.select(b2)}).rename("bsi")
    return img.addBands([ndvi, ndbi, ndwi, bsi])


def build_collection():
    def prep(image):
        image = mask_clouds(image)
        st = to_kelvin(image)
        # Landsat Collection 2 Level-2 surface reflectance scale and offset.
        # The additive offset must be applied before normalized-difference indices.
        sr = image.select(list(SR_BANDS.values())).multiply(0.0000275).add(-0.2)
        sr = sr.rename(list(SR_BANDS.keys()))
        out = sr.addBands([st])
        # map() 后显式保留时间戳，供 calendarRange 季节筛选使用。
        return add_indices(out).copyProperties(image, ["system:time_start"])

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    col = l8.merge(l9).filterDate(START, END).filterBounds(ee.Geometry.BBox(*BBOX))
    return col.map(prep)


def two_stage_composite(col, months):
    """先逐年做季节中位数，再对12个年度季节合成取中位数。

    该两级归约避免把约500景影像一次性送入区域归约而触发 GEE 用户内存上限；
    同时令每个年份对最终气候态季节合成具有相同权重。count 为各年度有效观测数之和。
    """
    annual = []
    for year in YEARS:
        yc = (col.filter(ee.Filter.calendarRange(year, year, "year"))
                .filter(ee.Filter.calendarRange(months[0], months[2], "month")))
        med = yc.select(OUTPUT_BANDS).median()
        cnt = yc.select("ST").reduce(ee.Reducer.count()).rename("count")
        annual.append(med.addBands(cnt).set("year", year))
    annual_col = ee.ImageCollection.fromImages(annual)
    science = annual_col.select(OUTPUT_BANDS).median()
    count = annual_col.select("count").sum().rename("count")
    return science.addBands(count)


def main():
    ee.Initialize(project=PROJECT)
    os.makedirs(OUT_DIR, exist_ok=True)
    seasons = sys.argv[1:] or list(SEASONS)
    aoi = ee.Geometry.BBox(*BBOX)
    col = build_collection()
    print("GEE 连接 OK，处理：", seasons)

    # 研究区按纬度切 6 片下载（单片更小，避免 getPixels 体积限制）
    mlo, mhi = BBOX[1], BBOX[3]
    n_tiles = 12
    step = (mhi - mlo) / n_tiles
    tiles = []
    for i in range(n_tiles):
        tlo = mlo + i * step
        thi = mlo + (i + 1) * step
        tiles.append((BBOX[0], tlo, BBOX[2], thi))

    for sname in seasons:
        months = SEASONS[sname]
        s_col = col.filter(ee.Filter.calendarRange(months[0], months[2], "month"))
        n = s_col.size().getInfo()
        print(f"\n[{sname}] 场景数: {n}")
        if n == 0:
            continue
        staged = two_stage_composite(col, months)
        # count 的无覆盖像元应为0，而不是科学波段使用的 -9999。
        comp = staged.select(OUTPUT_BANDS).unmask(-9999).addBands(
            staged.select("count").unmask(0))

        band_list = ["ST", "red", "nir08", "swir16",
                     "ndvi", "ndbi", "ndwi", "bsi", "count"]
        import requests
        for ti, tb in enumerate(tiles):
            out = os.path.join(OUT_DIR, f"gee_{sname}_tile{ti}.tif")
            if os.path.exists(out):
                print(f"  [{sname}] tile{ti} 已存在，跳过（续传）")
                continue
            ok = False
            for attempt in range(1, 4):   # 每片最多重试 3 次
                try:
                    url = comp.getDownloadURL({
                        "scale": SCALE, "crs": TARGET_CRS,
                        "region": ee.Geometry.BBox(*tb), "format": "GEO_TIFF",
                        "bands": band_list,
                    })
                    print(f"  下载 [{sname}] tile{ti} ({tb[3]:.2f}-{tb[1]:.2f}N) 第{attempt}次 ...")
                    t0 = time.time()
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    with open(out, "wb") as f:
                        f.write(r.content)
                    print(f"  ✅ 已保存 {out}（{os.path.getsize(out)/1e6:.1f}MB，{time.time()-t0:.0f}s）")
                    ok = True
                    break
                except Exception as e:
                    print(f"  第{attempt}次失败: {type(e).__name__} {str(e)[:90]}")
                    response = getattr(e, "response", None)
                    if response is not None and getattr(response, "text", None):
                        print(f"    GEE响应: {response.text[:500]}")
                    time.sleep(5 * attempt)
            if not ok:
                print(f"  ⚠️ [{sname}] tile{ti} 3 次都失败，跳过（可重跑续传）")


if __name__ == "__main__":
    main()
