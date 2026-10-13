#!/usr/bin/env bash
# Telephix 数据标注 — 启动脚本。
# 用法: bash run.sh   (第一次运行会自动调用 install.sh 装好环境)
# 端口被占用时: TELEPHIX_ANNOTATION_PORT=8098 bash run.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"    # 登录信息存储 .nicegui/ 固定放在项目目录里

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c 'import nicegui' 2>/dev/null; then
    echo "首次运行,先安装环境…"
    bash install.sh
fi

echo "启动 Telephix 数据标注 → http://localhost:${TELEPHIX_ANNOTATION_PORT:-8097}"
exec .venv/bin/python app.py
