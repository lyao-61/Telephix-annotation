#!/usr/bin/env bash
# Telephix 数据标注 — 新电脑一次性安装:系统工具 + Python 虚拟环境(.venv)。
# 用法: bash install.sh      (run.sh 发现没装好时会自动调用)
# 国内网络慢可换 pip 源: PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash install.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"

apt_install() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "❌ 不是 apt 系统,请手动安装: $*,然后重跑 bash install.sh" >&2
        exit 1
    fi
    echo "→ 安装系统工具: $*(需要 sudo 密码)"
    sudo apt-get update
    sudo apt-get install -y "$@"
}

# ── 1. 系统工具 ──
#   zenity  : 「选择文件夹」对话框
#   rsync   : 传输 session
#   sshpass : 传输时自动填对方密码
missing=()
command -v "$PY" >/dev/null 2>&1 || missing+=(python3)
for t in zenity rsync sshpass; do
    command -v "$t" >/dev/null 2>&1 || missing+=("$t")
done
[ ${#missing[@]} -eq 0 ] || apt_install "${missing[@]}"

"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || {
    echo "❌ 需要 Python ≥ 3.10,当前: $("$PY" --version 2>&1)" >&2
    exit 1
}

# ── 2. Python 虚拟环境 ──
if [ ! -x .venv/bin/python ]; then
    echo "→ 创建虚拟环境 .venv"
    if ! "$PY" -m venv .venv; then
        rm -rf .venv
        apt_install python3-venv          # Ubuntu/Debian 默认不带 venv
        "$PY" -m venv .venv
    fi
fi

echo "→ 安装 Python 依赖 (requirements.txt)"
.venv/bin/python -m pip install -r requirements.txt

echo "✅ 安装完成,启动: bash run.sh"
