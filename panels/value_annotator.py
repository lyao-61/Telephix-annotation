#!/usr/bin/env python3
"""Segment value annotator for int / float / bool annotation fields.

Subtask-style: assign a (type-restricted) value at the current video time
("在当前时间添加")，该值从这一时刻起生效直到下一个分段；或一键「采用整段视频」
把一个值套用到整段（= 只留一个 start=0 的分段）。

Own module (guide §14bis); the Annotate page's int/float/bool tabs host it
via panels/project_annotate.py. Per episode + per field it writes
``<episode>/<field>.json`` (next to subtasks.json, no ann_ prefix):
    { "field": "<name>", "type": "int|float|bool",
      "segments": [ {"start": <seconds>, "value": <typed>} ] }   # start 升序

Independent of existing annotate code. Needs only:
  • get_episode_dir() -> Path | None
  • video_selector    : CSS selector of the master <video> (read currentTime)
"""
from __future__ import annotations

import json

from nicegui import ui


def build_value_annotator(field: dict, get_episode_dir, video_selector: str,
                          on_saved=None):
    """Render the value annotator for one int/float/bool field. Returns a
    `reload()` callback the host calls on episode change.

    `on_saved` (optional): called after a successful save so the host can
    refresh the episode-dropdown progress badges."""
    name = field.get('name', 'value')
    typ = field.get('type', 'string')          # 'int' | 'float' | 'bool'
    st = {'segments': []}                       # [ {start: float, value} ]

    def _episode_dir():
        try:
            return get_episode_dir()
        except Exception:
            return None

    def _sidecar():
        ed = _episode_dir()
        # Sidecar sits next to subtasks.json, named "<field>.json" (no ann_ prefix).
        return (ed / f'{name}.json') if ed else None

    async def _current_time():
        try:
            v = await ui.run_javascript(
                f"const v=document.querySelector('{video_selector}');"
                "return v ? v.currentTime : null;", timeout=5.0)
            return float(v) if v is not None else None
        except Exception:
            return None

    # ── UI ────────────────────────────────────────────────────────────
    tlabel = {'int': '整数', 'float': '浮点数', 'bool': '布尔'}.get(typ, typ)
    ui.label(f'分段标注 · {name}（{tlabel}）—— 在某时刻设值，生效至下一分段；'
             '或「采用整段视频」把一个值套用到整段') \
        .style('font-size:14px; color:var(--text-2); margin-bottom:6px;')

    with ui.row().style('gap:10px; align-items:center; flex-wrap:wrap;'):
        if typ == 'bool':
            vwidget = ui.select({True: 'true', False: 'false'}, value=True, label='值') \
                .props('outlined dense').style('min-width:150px;')

            def _read():
                return bool(vwidget.value)
        elif typ == 'int':
            vwidget = ui.number('值', value=0, precision=0, step=1) \
                .props('outlined dense').style('min-width:150px;')

            def _read():
                return int(round(vwidget.value or 0))
        else:  # float
            vwidget = ui.number('值', value=0.0, step=0.1) \
                .props('outlined dense').style('min-width:150px;')

            def _read():
                return float(vwidget.value or 0.0)

        ui.button('在当前时间添加', icon='add_location_alt', on_click=lambda: _add_cursor()) \
            .props('unelevated color="primary" size="sm"').style('text-transform:none;')
        ui.button('采用整段视频', icon='select_all', on_click=lambda: _whole_video()) \
            .props('outline color="primary" size="sm"').style('text-transform:none;') \
            .tooltip('整段视频套用这一个值（清掉其它分段）')
        ui.element('div').style('flex:1;')
        ui.button(f'保存 {name}.json', icon='save', on_click=lambda: _save()) \
            .props('unelevated color="primary" size="sm"').style('text-transform:none;')

    ui.label('分段（start 升序，各值生效至下一分段）：') \
        .style('font-size:12px; color:var(--text-3); margin-top:8px;')
    seg_col = ui.column().style('width:100%; gap:4px;')

    # ── logic ─────────────────────────────────────────────────────────
    @ui.refreshable
    def _render():
        seg_col.clear()
        with seg_col:
            if not st['segments']:
                ui.label('（无）').style('font-size:12px; color:var(--text-3);')
            for i, s in enumerate(st['segments']):
                with ui.row().style('gap:8px; align-items:center; width:100%;'):
                    tag = '整段' if (len(st['segments']) == 1 and s['start'] <= 0.001) else f't={s["start"]:.2f}s'
                    ui.label(f'{tag}  →  {s["value"]}').style('font-size:13px; flex:1;')

                    def _del(idx=i):
                        st['segments'].pop(idx)
                        _render.refresh()
                    ui.button(icon='close', on_click=lambda _, f=_del: f()) \
                        .props('flat round dense size="sm" color="negative"')
    _render()

    async def _add_cursor():
        if not _episode_dir():
            ui.notify('请先在上方选择一个 episode', type='warning')
            return
        t = await _current_time()
        if t is None:
            ui.notify('读不到视频时间，请先让上方视频加载/暂停到目标时刻', type='warning')
            return
        v = _read()
        # 同一时刻覆盖
        st['segments'] = [s for s in st['segments'] if abs(s['start'] - t) > 1e-3]
        st['segments'].append({'start': round(float(t), 3), 'value': v})
        st['segments'].sort(key=lambda s: s['start'])
        _render.refresh()

    def _whole_video():
        if not _episode_dir():
            ui.notify('请先在上方选择一个 episode', type='warning')
            return
        st['segments'] = [{'start': 0.0, 'value': _read()}]
        _render.refresh()
        ui.notify('已设为整段视频一个值', type='positive')

    def _save():
        sc = _sidecar()
        if not sc:
            ui.notify('未选择 episode，无法保存', type='negative')
            return
        data = {'field': name, 'type': typ,
                'segments': sorted(st['segments'], key=lambda s: s['start'])}
        try:
            tmp = sc.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(sc)
        except Exception as ex:
            ui.notify(f'保存失败：{ex}', type='negative')
            return
        ui.notify(f'已保存 {len(data["segments"])} 个分段到 {name}.json', type='positive')
        if callable(on_saved):
            try:
                on_saved()
            except Exception:
                pass

    def reload():
        st['segments'] = []
        sc = _sidecar()
        if sc and sc.is_file():
            try:
                d = json.loads(sc.read_text(encoding='utf-8'))
                segs = d.get('segments') if isinstance(d, dict) else None
                if isinstance(segs, list):
                    st['segments'] = [
                        {'start': float(s.get('start', 0)), 'value': s.get('value')}
                        for s in segs if isinstance(s, dict)]
                    st['segments'].sort(key=lambda s: s['start'])
            except Exception:
                pass
        _render.refresh()

    return reload
