#!/usr/bin/env python
"""
阶段2（数据可得性盘点）：用 Microsoft Planetary Computer (MPC) STAC 搜索研究区
Landsat 8/9 Collection 2 Level-2 场景，统计「年 × 季节」的场景数、云量。

免登录、无账号门槛。输出：
  1. 控制台：按「年 × 季节」场景数 + 各季云量中位数。
  2. CSV：01_data/external/landsat_scene_inventory_mpc.csv（完整场景清单，供后续下载用）。

用法：
  .venv/bin/python 02_scripts/mpc/00_mpc_stac_inventory.py
"""
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

# 研究区：新墨西哥 Rio Grande Rift 中央裂谷带（D7）
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -108.0, 33.0, -106.0, 36.5
START, END = "2014-01-01", "2025-12-31"
COLLECTION = "landsat-c2-l2"
MAX_SCENES = 200000


def season_of(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def search_page(page: dict) -> list:
    req = urllib.request.Request(
        STAC, data=json.dumps(page).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    print(f"== MPC STAC 搜索 {COLLECTION}（{LON_MIN}~{LON_MAX}E, {LAT_MIN}~{LAT_MAX}N, {START}~{END}）==")
    payload = {
        "collections": [COLLECTION],
        "bbox": [LON_MIN, LAT_MIN, LON_MAX, LAT_MAX],
        "datetime": f"{START}T00:00:00Z/{END}T23:59:59Z",
        "limit": 500,
    }

    rows = []
    total = 0
    while True:
        out = search_page(payload)
        feats = out.get("features", [])
        for f in feats:
            p = f["properties"]
            rows.append({
                "scene_id": p.get("landsat:scene_id", f.get("id", "")),
                "item_id": f.get("id", ""),
                "acquisition_date": p.get("datetime", "")[:10],
                "cloud_cover": p.get("eo:cloud_cover"),
                "path": p.get("landsat:wrs_path", ""),
                "row": p.get("landsat:wrs_row", ""),
                "platform": p.get("platform", ""),
                "correction": p.get("landsat:correction", ""),
            })
        total += len(feats)
        # 分页
        links = out.get("links", [])
        nxt = next((l for l in links if l.get("rel") == "next"), None)
        if not nxt or total >= MAX_SCENES:
            break
        payload = nxt.get("body", payload)  # MPC 的 next 返回 body 带新 token
        print(f"  ... 已累计 {total} 条")

    print(f"\n命中场景数：{total}")
    if not rows:
        print("无场景，请检查范围/时段。")
        sys.exit(1)

    # 去除重复（同一 scene_id 可能跨瓦片重复）
    seen, uniq = set(), []
    for r in rows:
        k = (r["scene_id"], r["path"], r["row"], r["acquisition_date"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq

    # 年 × 季节 统计
    stat = defaultdict(lambda: {"n": 0, "clouds": []})
    for r in rows:
        try:
            dt = datetime.strptime(r["acquisition_date"], "%Y-%m-%d")
        except ValueError:
            continue
        key = (dt.year, season_of(dt.month))
        stat[key]["n"] += 1
        try:
            stat[key]["clouds"].append(float(r["cloud_cover"]))
        except (TypeError, ValueError):
            pass

    print(f"\n== 按「年 × 季节」场景数（及云量中位数%）==")
    print(f"{'年':>6} {'spring':>7} {'summer':>7} {'autumn':>7} {'winter':>7}   {'云量中位%':>9}")
    years = sorted({k[0] for k in stat})
    for y in years:
        spr = stat[(y, "spring")]["n"]
        smr = stat[(y, "summer")]["n"]
        aut = stat[(y, "autumn")]["n"]
        win = stat[(y, "winter")]["n"]
        allc = [c for k in stat if k[0] == y for c in stat[k]["clouds"]]
        med = round(sorted(allc)[len(allc) // 2], 1) if allc else float("nan")
        print(f"{y:>6} {spr:>7} {smr:>7} {aut:>7} {win:>7}   {med:>9}")

    # 传感器分布
    print("\n== 传感器分布 ==")
    for k, v in defaultdict(int, {a: sum(1 for r in rows if r["platform"] == a) for a in set(r["platform"] for r in rows)}).items():
        print(f"  {k}: {v}")

    out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "01_data", "external",
                           "landsat_scene_inventory_mpc.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ 已保存：{out_csv}（{len(rows)} 条唯一场景）")


if __name__ == "__main__":
    main()
