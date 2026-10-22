#!/usr/bin/env bash
# 阶段1：下载公开断层数据（均无需注册）
# 用法：bash 02_scripts/gee/00_download_fault_labels.sh
set -euo pipefail

# 回到项目根目录
cd "$(dirname "$0")/../.."
OUT=01_data/external
mkdir -p "$OUT"
cd "$OUT"

echo "== [1/2] 下载 GEM 中国活动断层（CAFD 派生，Zenodo，约 1.6 MB）=="
curl -L --fail --retry 2 -o chn_faults.geojson \
  "https://zenodo.org/records/7268805/files/chn_faults.geojson?download=1"
echo "   -> chn_faults.geojson  $(du -h chn_faults.geojson | cut -f1)"

echo ""
echo "== [2/2] 下载 GEM 全球活动断层（GitHub，交叉验证用）=="
if [ ! -d gem-global-active-faults ]; then
  git clone --depth 1 https://github.com/GEMScienceTools/gem-global-active-faults.git
else
  echo "   gem-global-active-faults 已存在，跳过"
fi

echo ""
echo "== 完成，01_data/external 当前内容 =="
ls -lh

echo ""
echo "== 可选：尝试 CAFD 官方 WFS 服务（1:40 万，最权威，验证是否免认证）=="
echo "   GetCapabilities 探测："
echo "   curl 'http://data.activetectonics.cn/arcserver/services/Hosted/CAFD400_2022_WFS/MapServer/WFSServer?service=WFS&request=GetCapabilities'"
