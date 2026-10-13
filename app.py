#!/usr/bin/env python3
"""Telephix 数据标注 — 独立标注工具入口.

    python3 app.py        →  http://localhost:8097

自包含,不依赖 Telephix-teleop / ROS。功能:登录 → 定义标注方案(字段) →
选数据文件夹标注(3 路视频 + 时间轴 + 接触模式 + subtask + 检测框/数值)→
写出 sidecar(subtasks.json / contact_modes.json / <field>.json / bbox.json)。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nicegui import ui  # noqa: E402

import pages.login       # noqa: E402,F401  registers '/'
import pages.projects    # noqa: E402,F401  registers '/projects'
import pages.annotate    # noqa: E402,F401  registers '/annotate'

ui.run(
    title='Telephix 数据标注',
    port=int(os.environ.get('TELEPHIX_ANNOTATION_PORT', '8097')),
    show=False,
    reload=False,
    dark=False,
    storage_secret='telephix-annotation-2026',
)
