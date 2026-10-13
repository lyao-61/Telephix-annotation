#!/usr/bin/env bash
# 打包成 dist/<项目名>-<日期>.tar.gz,拷到新电脑解压后 bash run.sh 即可。
# 不打包: .venv/(新电脑自己装)、.nicegui/(本机登录信息)、dist/、__pycache__
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="$(basename "$HERE")"
OUT="$HERE/dist/${NAME}-$(date +%Y%m%d).tar.gz"
mkdir -p "$HERE/dist"
tar -czf "$OUT" -C "$(dirname "$HERE")" \
    --exclude="$NAME/.venv" \
    --exclude="$NAME/.nicegui" \
    --exclude="$NAME/dist" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.tmp' \
    "$NAME"
echo "已打包: $OUT ($(du -h "$OUT" | cut -f1))"
