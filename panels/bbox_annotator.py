#!/usr/bin/env python3
"""Bounding-box (detection) annotator — draw boxes on a grabbed video frame
and label each object. Own module (guide §14); the Annotate page's
`detection`-type tab hosts it via panels/project_annotate.py.

Per episode it writes ``<episode>/bbox.json`` keyed by frame number:
    { "field": "<name>", "fps": 30.0,
      "frames": { "<frame>": [ {"bbox": [x,y,w,h] (归一化), "label": "..."} ] } }

Independent of existing annotate code. Needs only:
  • get_episode_dir()  -> Path | None   (current Annotate episode)
  • video_selector     : CSS selector for the master <video> (JS frame grab)
Object labels are free-input with an MRU "Recent" dropdown (like subtask).
Frame image is grabbed by pausing the video and reading its current frame.
"""
from __future__ import annotations

import json

from nicegui import ui


def _esc(s: str) -> str:
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def build_bbox_annotator(field: dict, get_episode_dir, video_selector: str,
                         on_saved=None):
    """Render the detection annotator for one field. Returns a `reload()`
    callback the host calls when the Annotate episode changes.

    `on_saved` (optional): called after a successful save so the host can
    refresh the episode-dropdown progress badges."""
    name = field.get('name', 'objects')
    st = {
        'frame': None,       # int frame index of the grabbed frame
        'w': 0, 'h': 0,      # grabbed frame pixel size
        'start': None,       # drag start (image px) between mousedown/up
        'pending': None,     # [x,y,w,h] normalized, drawn dashed until labeled
        'by_frame': {},      # {frame:int -> [ {bbox,label} ]}
        'recent': [],        # MRU labels
        'fps': 30.0,
    }

    def _episode_dir():
        try:
            return get_episode_dir()
        except Exception:
            return None

    def _sidecar():
        ed = _episode_dir()
        return (ed / 'bbox.json') if ed else None

    def _fps() -> float:
        ed = _episode_dir()
        try:
            m = json.loads((ed / 'meta.json').read_text(encoding='utf-8'))
            return float(m.get('fps') or 30.0)
        except Exception:
            return 30.0

    # ── UI ────────────────────────────────────────────────────────────
    ui.label('检测框标注（在画面上拖拽画框，松手后填/选物体类别）') \
        .style('font-size:14px; color:var(--text-2); margin-bottom:6px;')

    img = ui.interactive_image('', events=['mousedown', 'mouseup'], cross=True,
                               on_mouse=lambda e: _on_mouse(e)) \
        .style('width:100%; max-width:760px; border-radius:8px; background:#000;')

    with ui.row().style('gap:10px; align-items:center; flex-wrap:wrap; margin-top:8px;'):
        lbl_input = ui.input('物体类别（拖框后填写）') \
            .props('outlined dense clearable').style('min-width:220px;')
        recent_sel = ui.select([], value=None, label='Recent') \
            .props('outlined dense clearable options-dense').style('min-width:170px;')
        recent_sel.on_value_change(
            lambda: lbl_input.set_value(recent_sel.value) if recent_sel.value else None)
        ui.button('添加框', icon='add', on_click=lambda: _commit()) \
            .props('unelevated color="primary" size="sm"').style('text-transform:none;')

    ui.label('本帧已标注：').style('font-size:12px; color:var(--text-3); margin-top:8px;')
    boxes_col = ui.column().style('width:100%; gap:4px;')

    # 抓取当前帧 / 保存 bbox.json —— 放在最底部,紧挨 episode 导航上方
    with ui.row().style('gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; '
                        'padding-top:10px; border-top:1px solid rgba(0,0,0,0.06);'):
        ui.button('抓取当前帧', icon='photo_camera', on_click=lambda: _grab()) \
            .props('unelevated color="primary" size="sm"').style('text-transform:none;')
        frame_label = ui.label('未抓取（先把上方视频暂停到目标帧）') \
            .style('font-size:13px; color:var(--text-3);')
        ui.element('div').style('flex:1;')
        ui.button('保存 bbox.json', icon='save', on_click=lambda: _save()) \
            .props('unelevated color="primary" size="sm"').style('text-transform:none;')

    # ── logic ─────────────────────────────────────────────────────────
    def _cur_boxes():
        return st['by_frame'].setdefault(st['frame'], []) if st['frame'] is not None else []

    @ui.refreshable
    def _render():
        # 1) SVG overlay on the frame (committed = solid, pending = dashed)
        W, H = st['w'] or 1, st['h'] or 1
        svg = []
        for b in (_cur_boxes() if st['frame'] is not None else []):
            x, y, bw, bh = b['bbox']
            svg.append(f'<rect x="{x*W:.0f}" y="{y*H:.0f}" width="{bw*W:.0f}" '
                       f'height="{bh*H:.0f}" fill="none" stroke="#00E5FF" stroke-width="3"/>')
            svg.append(f'<text x="{x*W+5:.0f}" y="{y*H+24:.0f}" fill="#00E5FF" '
                       f'font-size="20" font-weight="bold">{_esc(b["label"])}</text>')
        if st['pending']:
            x, y, bw, bh = st['pending']
            svg.append(f'<rect x="{x*W:.0f}" y="{y*H:.0f}" width="{bw*W:.0f}" '
                       f'height="{bh*H:.0f}" fill="rgba(255,193,7,0.12)" '
                       f'stroke="#FFC107" stroke-width="3" stroke-dasharray="7"/>')
        img.set_content(''.join(svg))
        # 2) list of this frame's boxes
        boxes_col.clear()
        with boxes_col:
            boxes = _cur_boxes() if st['frame'] is not None else []
            if not boxes:
                ui.label('（无）').style('font-size:12px; color:var(--text-3);')
            for i, b in enumerate(boxes):
                with ui.row().style('gap:8px; align-items:center; width:100%;'):
                    xywh = ', '.join(f'{v:.3f}' for v in b['bbox'])
                    ui.label(f'{b["label"]}  [{xywh}]').style('font-size:13px; flex:1;')

                    def _del(idx=i):
                        boxes.pop(idx)
                        _render.refresh()
                    ui.button(icon='close', on_click=lambda _, f=_del: f()) \
                        .props('flat round dense size="sm" color="negative"')
    _render()

    def _on_mouse(e):
        if st['frame'] is None:
            return
        if e.type == 'mousedown':
            st['start'] = (e.image_x, e.image_y)
        elif e.type == 'mouseup' and st['start'] is not None:
            x0, y0 = st['start']
            x1, y1 = e.image_x, e.image_y
            st['start'] = None
            W, H = st['w'] or 1, st['h'] or 1
            nx, ny = min(x0, x1) / W, min(y0, y1) / H
            nw, nh = abs(x1 - x0) / W, abs(y1 - y0) / H
            if nw < 0.005 or nh < 0.005:      # ignore stray clicks
                return
            st['pending'] = [round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)]
            _render.refresh()

    async def _grab():
        ed = _episode_dir()
        if not ed:
            ui.notify('请先在上方选择一个 episode', type='warning')
            return
        js = (
            f"const v = document.querySelector('{video_selector}');"
            "if (!v || !v.videoWidth) return null;"
            "const c = document.createElement('canvas');"
            "c.width = v.videoWidth; c.height = v.videoHeight;"
            "c.getContext('2d').drawImage(v, 0, 0);"
            "return {url: c.toDataURL('image/jpeg', 0.85), t: v.currentTime,"
            " w: v.videoWidth, h: v.videoHeight};"
        )
        try:
            res = await ui.run_javascript(js, timeout=8.0)
        except Exception as ex:
            ui.notify(f'抓帧失败：{ex}', type='negative')
            return
        if not res:
            ui.notify('抓帧失败：请先让上方视频加载/暂停到目标帧', type='warning')
            return
        st['fps'] = _fps()
        st['frame'] = int(round(float(res.get('t', 0)) * st['fps']))
        st['w'], st['h'] = int(res.get('w') or 0), int(res.get('h') or 0)
        st['pending'] = None
        st['by_frame'].setdefault(st['frame'], [])
        img.set_source(res['url'])
        frame_label.text = (f'帧 {st["frame"]}（t={float(res.get("t", 0)):.2f}s · '
                            f'{st["w"]}×{st["h"]}）')
        _render.refresh()

    def _commit():
        if not st['pending']:
            ui.notify('请先在画面上拖拽画一个框', type='warning')
            return
        label = (lbl_input.value or '').strip()
        if not label:
            ui.notify('请填写物体类别', type='warning')
            return
        _cur_boxes().append({'bbox': st['pending'], 'label': label})
        if label in st['recent']:
            st['recent'].remove(label)
        st['recent'].insert(0, label)
        del st['recent'][20:]
        recent_sel.set_options(list(st['recent']))
        st['pending'] = None
        lbl_input.set_value('')
        _render.refresh()

    def _save():
        sc = _sidecar()
        if not sc:
            ui.notify('未选择 episode，无法保存', type='negative')
            return
        data = {'field': name, 'fps': st['fps'],
                'frames': {str(k): v for k, v in st['by_frame'].items() if v}}
        total = sum(len(v) for v in st['by_frame'].values())
        try:
            tmp = sc.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(sc)
        except Exception as ex:
            ui.notify(f'保存失败：{ex}', type='negative')
            return
        ui.notify(f'已保存 {total} 个框（{len(data["frames"])} 帧）到 bbox.json',
                  type='positive')
        if callable(on_saved):
            try:
                on_saved()
            except Exception:
                pass

    def reload():
        st.update(frame=None, w=0, h=0, start=None, pending=None, by_frame={},
                  fps=_fps())
        st['recent'] = list(st['recent'])   # keep MRU across episodes
        sc = _sidecar()
        if sc and sc.is_file():
            try:
                d = json.loads(sc.read_text(encoding='utf-8'))
                fr = d.get('frames') if isinstance(d, dict) else None
                if isinstance(fr, dict):
                    st['by_frame'] = {int(k): v for k, v in fr.items()}
            except Exception:
                pass
        img.set_source('')
        img.set_content('')
        frame_label.text = '未抓取（先把上方视频暂停到目标帧）'
        _render.refresh()

    return reload
