#!/usr/bin/env python3
"""Project-spec annotation sub-modules for the Annotate page (own module).

Injected below the video display via a thin hook in app.py's Annotate
panel. Reads ``S.active_project['annotations']`` (defined when editing the
project, guide §12) and renders a horizontal nav — one entry per annotation
type — plus a content area per type. Content is a BLANK placeholder for now;
the real per-type annotation UIs land later (guide §13.7).

Self-contained: imports only nicegui + state; does NOT touch or reuse the
existing contact-mode / subtask code on the Annotate page.
"""
from __future__ import annotations

from nicegui import ui
from state import S

from panels.bbox_annotator import build_bbox_annotator
from panels.value_annotator import build_value_annotator


def build_project_annotate(ctx: dict | None = None) -> dict:
    """Render the annotation sub-module nav + per-type panels.

    Called inside app.py's Annotate panel, just below the video/TIMELINE.
    Reads the active project's annotation spec.

    `ctx` (optional) wires richer per-type UIs to the Annotate page:
      • ctx['get_episode_dir']()  -> current episode Path | None
      • ctx['video_selector']     : CSS selector of the master <video>
      • ctx['reloaders']          : list; this fn appends per-tab reload()
        callbacks the caller runs on episode change.
    `detection`-type fields host the bbox annotator (panels/bbox_annotator);
    others show a placeholder.

    Returns ``{field_name: tab_panel}`` so the caller can host extra UI in a
    specific field's panel (e.g. app.py moves the subtask UI into "action").
    """
    proj = getattr(S, 'active_project', None)
    anns = (proj or {}).get('annotations', []) or []
    panels: dict = {}

    with ui.element('div').style('width:100%; margin-top:18px;'):
        ui.html('<div style="font-size:11px; font-weight:700; opacity:0.5; '
                'letter-spacing:0.5px; margin:14px 0 6px;">'
                '标注子模块 · ANNOTATION</div>')

        if not proj:
            ui.label('未选择工程（从工程管理进入采集页后这里显示工程的标注项）') \
                .style('font-size:13px; color:var(--text-3);')
            return panels
        if not anns:
            ui.label('此工程未定义标注项 —— 在工程管理里编辑工程的「标注项」步骤添加') \
                .style('font-size:13px; color:var(--text-3);')
            return panels

        # Horizontal nav: one tab per annotation type (from project spec).
        first = anns[0]['id']
        with ui.tabs().props(
            'dense align=left active-color=primary indicator-color=primary'
        ).style('width:100%;') as tabs:
            for a in anns:
                req = ' *' if a.get('required') else ''
                ui.tab(a['id'],
                       label=f"{a.get('name', '?')}{req}  ·  {a.get('type', '?')}")

        with ui.tab_panels(tabs, value=first).style('width:100%; margin-top:8px;'):
            for a in anns:
                panel = ui.tab_panel(a['id'])
                with panel:
                    _t = a.get('type')
                    _has_ctx = bool(ctx and ctx.get('get_episode_dir'))
                    if _t == 'detection' and _has_ctx:
                        _reload = build_bbox_annotator(
                            a, ctx['get_episode_dir'],
                            ctx.get('video_selector', 'video'),
                            on_saved=ctx.get('on_saved'))
                        if isinstance(ctx.get('reloaders'), list):
                            ctx['reloaders'].append(_reload)
                    elif _t in ('int', 'float', 'bool') and _has_ctx:
                        _reload = build_value_annotator(
                            a, ctx['get_episode_dir'],
                            ctx.get('video_selector', 'video'),
                            on_saved=ctx.get('on_saved'))
                        if isinstance(ctx.get('reloaders'), list):
                            ctx['reloaders'].append(_reload)
                    else:
                        _placeholder(a)
                panels[a.get('name')] = panel
    return panels


def _placeholder(a: dict) -> None:
    """Blank placeholder panel for one annotation type (real UI: §13.7)."""
    name = a.get('name', '?')
    typ = a.get('type', '?')
    desc = (a.get('description') or '').strip()
    with ui.element('div').style(
        'width:100%; min-height:160px; border:1px dashed rgba(0,0,0,0.14); '
        'border-radius:10px; padding:22px; display:flex; flex-direction:column; '
        'gap:8px; align-items:flex-start; justify-content:center; '
        'background:rgba(0,0,0,0.015);'
    ):
        ui.html(
            f'<div style="font-size:16px; font-weight:700;">{name}'
            f'<span style="font-size:12px; font-weight:600; color:#007AFF; '
            f'background:rgba(0,122,255,0.08); padding:2px 9px; border-radius:10px; '
            f'margin-left:8px;">{typ}</span>'
            + ('<span style="font-size:12px; font-weight:600; color:#FF9500; '
               'background:rgba(255,149,0,0.10); padding:2px 9px; border-radius:10px; '
               'margin-left:6px;">必填</span>' if a.get('required') else '')
            + '</div>'
        )
        if desc:
            ui.label(desc).style('font-size:13px; color:var(--text-2);')
        if a.get('type') == 'enum' and a.get('options'):
            ui.label('选项：' + '、'.join(a['options'])) \
                .style('font-size:12px; color:var(--text-3);')
        ui.label(f'「{typ}」类型标注界面 —— 待实现（占位）') \
            .style('font-size:13px; color:var(--text-3); margin-top:4px;')
