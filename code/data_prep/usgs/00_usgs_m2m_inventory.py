#!/usr/bin/env python
"""
阶段2（数据可得性盘点）第一步：用 USGS M2M API 搜索研究区 Landsat 8/9 Collection 2 Level-2 场景。

输出：
  1. 控制台：按「年 × 季节」的场景数、云量中位数汇总。
  2. CSV：01_data/external/landsat_scene_inventory.csv（完整场景清单，供后续下载用）。

前置：
  - 注册 USGS EROS 账号：https://ers.cr.usgs.gov/register
  - 复制 03_configs/usgs_m2m.env.example 为 03_configs/usgs_m2m.env 并填入账号密码。

用法：
  .venv/bin/python 02_scripts/usgs/00_usgs_m2m_inventory.py
"""
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime

BASE = "https://m2m.cr.usgs.gov/api/api/json/stable/"

# 研究区：新墨西哥 Rio Grande Rift 中央裂谷带（D7）
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -108.0, 33.0, -106.0, 36.5
START, END = "2014-01-01", "2025-12-31"
DATASET = "landsat_ot_c2_l2"  # Landsat 8/9 OLI/TIRS Collection 2 Level 2（含地表温度 ST）

# 季节划分（北半球）
def season_of(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def load_creds():
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", "03_configs", "usgs_m2m.env")
    creds = {}
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    creds.setdefault("USGS_USERNAME", os.environ.get("USGS_USERNAME"))
    creds.setdefault("USGS_PASSWORD", os.environ.get("USGS_PASSWORD"))
    if not creds.get("USGS_USERNAME") or creds["USGS_USERNAME"].startswith("your_"):
        sys.exit("❌ 未配置 USGS 凭证。请先复制 03_configs/usgs_m2m.env.example 为 usgs_m2m.env 并填入账号密码。")
    return creds


def api(endpoint: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("X-Auth-Token", token)
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode("utf-8"))
    if out.get("errorCode"):
        raise RuntimeError(f"{endpoint} 错误 {out['errorCode']}: {out.get('errorMessage')}")
    return out.get("data")


def meta_to_dict(result: dict) -> dict:
    """把 M2M 返回的 metadata 列表转成 {fieldName: value}。"""
    d = {}
    for m in result.get("metadata", []):
        name = m.get("fieldName")
        if name:
            d[name] = m.get("value")
    return d


def main():
    creds = load_creds()
    print(f"== 登录 USGS M2M（{creds['USGS_USERNAME']}）==")
    token = api("login", {"username": creds["USGS_USERNAME"], "password": creds["USGS_PASSWORD"]})
    try:
        print(f"== 搜索 {DATASET} 场景：{LON_MIN}~{LON_MAX}E, {LAT_MIN}~{LAT_MAX}N, {START}~{END} ==")
        payload = {
            "datasetName": DATASET,
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": LAT_MIN, "longitude": LON_MIN},
                "upperRight": {"latitude": LAT_MAX, "longitude": LON_MAX},
            },
            "temporalFilter": {"start": START, "end": END},
            "maxResults": 50000,
        }
        data = api("scene-search", payload, token)
        results = data.get("results", []) or []
        print(f"命中场景数：{data.get('totalHits', len(results))}")

        rows = []
        for r in results:
            m = meta_to_dict(r)
            acq = m.get("Acquisition Date") or ""
            cloud = m.get("Cloud Cover")
            rows.append({
                "display_id": m.get("Display ID", r.get("displayId", "")),
                "entity_id": m.get("Entity ID", r.get("entityId", "")),
                "acquisition_date": acq,
                "cloud_cover": cloud,
                "path": m.get("WRS Path", ""),
                "row": m.get("WRS Row", ""),
                "sensor": m.get("Sensor Identifier", ""),
            })

        # 季节/年份统计
        from collections import defaultdict
        stat = defaultdict(lambda: {"n": 0, "clouds": []})
        for row in rows:
            try:
                dt = datetime.strptime(row["acquisition_date"][:10], "%Y-%m-%d")
            except ValueError:
                continue
            key = (dt.year, season_of(dt.month))
            stat[key]["n"] += 1
            try:
                stat[key]["clouds"].append(float(row["cloud_cover"]))
            except (TypeError, ValueError):
                pass

        print("\n== 按「年 × 季节」场景数（及云量中位数%）==")
        print(f"{'年':>6} {'春':>4} {'夏':>4} {'秋':>4} {'冬':>4}   {'云量中位%':>8}")
        years = sorted({k[0] for k in stat})
        for y in years:
            spr = stat[(y, "spring")]["n"]
            smr = stat[(y, "summer")]["n"]
            aut = stat[(y, "autumn")]["n"]
            win = stat[(y, "winter")]["n"]
            allc = [c for k in stat if k[0] == y for c in stat[k]["clouds"]]
            med = round(sorted(allc)[len(allc) // 2], 1) if allc else float("nan")
            print(f"{y:>6} {spr:>4} {smr:>4} {aut:>4} {win:>4}   {med:>8}")

        # 保存清单
        out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "01_data", "external",
                               "landsat_scene_inventory.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        if rows:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\n✅ 已保存场景清单：{out_csv}（{len(rows)} 条）")
    finally:
        api("logout", {}, token)


if __name__ == "__main__":
    main()
