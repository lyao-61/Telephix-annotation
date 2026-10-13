"""Login ('/') — capture the collector's and annotator's names (stored
per-browser; the annotator is written into every sidecar's ``annotator`` field,
both are added to the folder name on transfer)."""
from __future__ import annotations

from nicegui import app, ui
from styles import CSS


@ui.page('/')
def login_page():
    u = app.storage.user
    if u.get('username') and u.get('collector'):
        ui.navigate.to('/projects')
        return
    ui.add_css(CSS)
    ui.html('<div class="ambient"></div>')
    with ui.element('div').style(
        'position:relative; z-index:1; min-height:100vh; display:flex; '
        'align-items:center; justify-content:center;'
    ):
        with ui.element('div').classes('glass').style(
            'padding:44px 48px; display:flex; flex-direction:column; '
            'gap:16px; width:380px;'
        ):
            ui.html('<div style="font-size:26px; font-weight:800;">Telephix '
                    '<span style="opacity:0.45; font-size:18px;">数据标注</span></div>')
            ui.label('输入采集员和标注员名字即可开始;标注员会记录到 annotator 字段,'
                     '传输时两个名字都会加进文件夹名。') \
                .style('font-size:13px; color:var(--text-3);')
            collector = ui.input('采集员名字', value=u.get('collector', '')) \
                .props('outlined autofocus').style('width:100%;')
            name = ui.input('标注员名字', value=u.get('username', '')) \
                .props('outlined').style('width:100%;')

            def _go():
                c = (collector.value or '').strip()
                v = (name.value or '').strip()
                if not c:
                    ui.notify('请输入采集员名字', type='warning')
                    return
                if not v:
                    ui.notify('请输入标注员名字', type='warning')
                    return
                u['collector'] = c
                u['username'] = v
                ui.navigate.to('/projects')
            collector.on('keydown.enter', lambda: _go())
            name.on('keydown.enter', lambda: _go())
            ui.button('进入', icon='login', on_click=_go) \
                .props('unelevated color="primary"') \
                .style('text-transform:none; font-weight:600; border-radius:12px;')
