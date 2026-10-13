"""标注方案 ('/projects') — list / create / edit annotation schemas, then pick
one to annotate. A schema = a set of annotation fields (name + type)."""
from __future__ import annotations

from nicegui import app, ui

from auth import require_login
from styles import CSS
from schema_store import (
    ANN_TYPES,
    delete_project,
    load_projects,
    save_project,
)


@ui.page('/projects')
def projects_page():
    if not require_login():
        return
    ui.add_css(CSS)
    ui.html('<div class="ambient"></div>')

    # ── one schema card ──
    def _card(p, grid_ref):
        anns = p.get('annotations', []) or []
        with ui.element('div').classes('glass').style(
            'padding:26px 28px; display:flex; flex-direction:column; gap:10px;'
        ):
            ui.label(p.get('name', '')).style('font-size:22px; font-weight:800;')
            ui.label(p.get('id', '')).style(
                'font-size:12px; color:var(--text-3); font-family:monospace; '
                'background:rgba(0,0,0,0.045); padding:3px 10px; border-radius:6px; '
                'align-self:flex-start; user-select:all;')
            if anns:
                chips = ' '.join(
                    f'<span style="display:inline-block; font-size:12px; '
                    f'background:rgba(0,122,255,0.09); color:#2a78d6; '
                    f'padding:2px 9px; border-radius:12px; margin:2px 4px 2px 0;">'
                    f'{a.get("name", "")} · {a.get("type", "")}</span>'
                    for a in anns)
                ui.html(f'<div>{chips}</div>')
            else:
                ui.label('（无标注字段，点编辑添加）') \
                    .style('font-size:13px; color:var(--text-3);')
            with ui.row().style('margin-top:8px; gap:8px; align-items:center; '
                                'width:100%;'):
                ui.button('进入标注', icon='login',
                          on_click=lambda _=None, pp=p: _enter(pp)) \
                    .props('unelevated color="primary"') \
                    .style('text-transform:none; font-weight:600; flex:1; '
                           'border-radius:12px;')
                ui.button(icon='edit',
                          on_click=lambda _=None, pp=p: _open_editor(pp, grid_ref)) \
                    .props('flat round color="grey-8"').tooltip('编辑')
                ui.button(icon='delete',
                          on_click=lambda _=None, pp=p: _confirm_delete(pp, grid_ref)) \
                    .props('flat round color="negative"').tooltip('删除')

    def _enter(p):
        if not (p.get('annotations') or []):
            ui.notify('该方案还没有标注字段，请先编辑添加', type='warning')
            return
        app.storage.user['active_project'] = p['id']
        ui.navigate.to('/annotate')

    def _confirm_delete(p, grid_ref):
        d = ui.dialog()
        with d, ui.card().style('padding:22px; gap:12px; min-width:340px;'):
            ui.label('删除这个标注方案？').style('font-size:16px; font-weight:700;')
            ui.label(p.get('name', '')).style('font-size:13px; color:var(--text-2);')
            ui.label('只删方案定义，不影响已写出的 sidecar 标注文件。') \
                .style('font-size:12px; color:var(--text-3);')
            with ui.row().style('justify-content:flex-end; gap:8px; width:100%;'):
                ui.button('取消', on_click=d.close).props('flat color="grey-8"') \
                    .style('text-transform:none;')

                def _do():
                    delete_project(p['id'])
                    d.close()
                    grid_ref.refresh()
                    ui.notify('已删除', type='info')
                ui.button('删除', on_click=_do).props('unelevated color="negative"') \
                    .style('text-transform:none;')
        d.open()

    def _open_editor(p, grid_ref=None):
        # working copy
        st = {'id': (p or {}).get('id'),
              'name': (p or {}).get('name', ''),
              'annotations': [dict(a) for a in ((p or {}).get('annotations') or [])],
              'next_ann_seq': (p or {}).get('next_ann_seq')}
        d = ui.dialog().props('persistent')
        with d, ui.card().style('width:640px; max-width:95vw; padding:26px; gap:14px; '
                                'max-height:88vh; overflow:auto;'):
            ui.label('编辑标注方案' if st['id'] else '新建标注方案') \
                .style('font-size:22px; font-weight:800;')
            e_name = ui.input('方案名称 *', value=st['name']) \
                .props('outlined').style('width:100%;') \
                .on_value_change(lambda e: st.__setitem__('name', e.value or ''))

            ui.html('<div style="font-size:12px; font-weight:800; letter-spacing:0.5px; '
                    'color:var(--text-3); margin-top:6px; text-transform:uppercase;">'
                    '标注字段 · FIELDS</div>')
            ui.label('字段名仅字母/数字/下划线;类型决定标注方式(string→subtask 时间轴,'
                     'detection→画框,int/float/bool→数值分段)。') \
                .style('font-size:12px; color:var(--text-3);')
            fields_box = ui.column().style('width:100%; gap:8px;')

            @ui.refreshable
            def _render_fields():
                fields_box.clear()
                with fields_box:
                    if not st['annotations']:
                        ui.label('还没有字段，点下方「+ 添加字段」') \
                            .style('font-size:13px; color:var(--text-3);')
                    for i, a in enumerate(st['annotations']):
                        with ui.row().style('gap:8px; align-items:center; width:100%; '
                                            'flex-wrap:nowrap;'):
                            ui.input('字段名', value=a.get('name', '')) \
                                .props('outlined dense').style('flex:1;') \
                                .on_value_change(
                                    lambda e, aa=a: aa.__setitem__('name', e.value or ''))
                            ui.select(ANN_TYPES, value=a.get('type', 'string'),
                                      label='类型') \
                                .props('outlined dense options-dense') \
                                .style('width:230px;') \
                                .on_value_change(
                                    lambda e, aa=a: aa.__setitem__('type', e.value))

                            def _del(idx=i):
                                st['annotations'].pop(idx)
                                _render_fields.refresh()
                            ui.button(icon='close', on_click=lambda _=None, f=_del: f()) \
                                .props('flat round dense size=sm color=grey-7')
            _render_fields()

            def _add_field():
                st['annotations'].append({'name': '', 'type': 'string'})
                _render_fields.refresh()
            ui.button('+ 添加字段', icon='add', on_click=_add_field) \
                .props('flat color="primary"').style('text-transform:none; '
                                                     'align-self:flex-start;')

            with ui.row().style('justify-content:flex-end; gap:8px; width:100%; '
                                'margin-top:6px;'):
                ui.button('取消', on_click=d.close).props('flat color="grey-8"') \
                    .style('text-transform:none;')

                def _save():
                    try:
                        save_project(st)
                    except ValueError as ex:
                        ui.notify(str(ex), type='negative')
                        return
                    d.close()
                    if grid_ref is not None:
                        grid_ref.refresh()
                    else:
                        grid.refresh()
                    ui.notify('已保存', type='positive')
                ui.button('保存', icon='save', on_click=_save) \
                    .props('unelevated color="primary"').style('text-transform:none;')
        d.open()

    with ui.element('div').style(
        'position:relative; z-index:1; display:flex; flex-direction:column; '
        'width:100%; min-height:100vh;'
    ):
        with ui.element('div').style(
            'display:flex; align-items:center; padding:12px 28px; gap:14px; '
            'background:rgba(255,255,255,0.8); backdrop-filter:blur(40px); '
            'border-bottom:1px solid rgba(0,0,0,0.06);'
        ):
            ui.html('<div style="font-size:20px; font-weight:800;">Telephix '
                    '<span style="opacity:0.45; font-size:15px;">数据标注 · 方案</span></div>')
            ui.element('div').style('flex:1;')
            ui.label(f'采集员：{app.storage.user.get("collector", "")} · '
                     f'标注员：{app.storage.user.get("username", "")}') \
                .style('font-size:13px; color:var(--text-3);')
            ui.button('新建方案', icon='add', on_click=lambda: _open_editor(None)) \
                .props('unelevated color="primary"').style('text-transform:none;')
            ui.button('退出登录', icon='logout', on_click=_logout) \
                .props('flat color="grey-8"').style('text-transform:none;')

        with ui.element('div').style('padding:26px 32px;'):
            @ui.refreshable
            def grid():
                projects = load_projects()
                if not projects:
                    with ui.column().style('width:100%; align-items:center; '
                                           'padding:60px 0; gap:8px;'):
                        ui.icon('label', size='52px').style('opacity:0.18;')
                        ui.label('还没有标注方案，点右上角「新建方案」').style(
                            'font-size:16px; font-weight:600; opacity:0.55;')
                    return
                with ui.element('div').style(
                    'display:grid; gap:24px; width:100%; '
                    'grid-template-columns:repeat(auto-fill, minmax(380px, 1fr));'
                ):
                    for p in projects:
                        _card(p, grid)
            grid()



def _logout():
    for k in ('username', 'collector', 'active_project'):
        app.storage.user.pop(k, None)
    ui.navigate.to('/')
