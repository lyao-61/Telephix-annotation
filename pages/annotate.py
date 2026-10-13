"""数据标注 ('/annotate') — hosts the full Annotate UI (3-cam video + timeline +
contact-mode C1–C6 + subtask + typed/detection annotators + episode nav). Reads
the active schema for the annotation fields, then delegates to build_annotate."""
from __future__ import annotations

from pathlib import Path

from nicegui import app, ui

from auth import require_login
from styles import CSS
from state import S
from schema_store import get_project
from dirpick import _pick_directory_dialog
from panels.annotate_stage import build_annotate
import rsync_transfer


@ui.page('/annotate')
def annotate_page():
    if not require_login():
        return
    pid = app.storage.user.get('active_project')
    proj = get_project(pid) if pid else None
    if not proj:
        ui.navigate.to('/projects')
        return
    S.active_project = proj    # annotate submodules read this for field defs

    ui.add_css(CSS)
    ui.html('<div class="ambient"></div>')

    with ui.element('div').style(
        'position:relative; z-index:1; display:flex; flex-direction:column; '
        'width:100%; max-width:100vw; height:100vh; overflow:hidden;'
    ):
        with ui.element('div').style(
            'display:flex; align-items:center; padding:10px 24px; gap:14px; '
            'flex-shrink:0; background:rgba(255,255,255,0.8); '
            'backdrop-filter:blur(40px); border-bottom:1px solid rgba(0,0,0,0.06);'
        ):
            ui.html('<div style="display:flex; align-items:baseline; gap:10px;">'
                    '<div style="font-size:20px; font-weight:800;">Telephix</div>'
                    '<div style="font-size:15px; font-weight:600; opacity:0.45;">'
                    '数据标注</div></div>')
            ui.html(f'<div style="font-size:14px; font-weight:700; color:#007AFF; '
                    f'background:rgba(0,122,255,0.08); padding:4px 14px; '
                    f'border-radius:16px;">{proj.get("name", "")}</div>')
            ui.element('div').style('flex:1;')
            ui.label(f'采集员：{app.storage.user.get("collector", "")} · '
                     f'标注员：{app.storage.user.get("username", "")}') \
                .style('font-size:13px; color:var(--text-3);')
            ui.button('返回方案', icon='arrow_back',
                      on_click=lambda: ui.navigate.to('/projects')) \
                .props('flat color="grey-8"').style('text-transform:none;')

        container = ui.element('div').style(
            'flex:1; overflow-y:auto; padding:16px 24px;')

        # ── 底部:传输本 session(rsync) ──
        _sess = {'dir': None}

        def _on_ep(ep):
            try:
                sd = Path(ep['episode_dir']).parent   # <root>/session_xxx_lerobot
            except Exception:
                return
            _sess['dir'] = str(sd)
            sess_lbl.set_text(f'当前 session：{sd.name}')
            sess_lbl.tooltip(str(sd))
            xfer_btn.props(remove='disable')

        with ui.element('div').style(
            'flex-shrink:0; display:flex; align-items:center; gap:14px; '
            'padding:10px 24px; background:rgba(255,255,255,0.85); '
            'backdrop-filter:blur(30px); border-top:1px solid rgba(0,0,0,0.08);'
        ):
            ui.icon('cloud_upload', size='22px').style('color:var(--blue);')
            sess_lbl = ui.label('当前 session：（选择 episode 后显示）') \
                .style('font-size:13px; color:var(--text-2);')
            ui.element('div').style('flex:1;')
            ui.label('rsync -avhP 把整个 session 文件夹传到另一台机器'
                     '（文件夹名自动加上采集员、标注员）') \
                .style('font-size:12px; color:var(--text-3);')
            xfer_btn = ui.button('传输本 session', icon='drive_file_move',
                                 on_click=lambda: _open_transfer()) \
                .props('unelevated color="teal" disable') \
                .style('text-transform:none; font-weight:600;')

        build_annotate(
            container,
            pick_directory_dialog=_pick_directory_dialog,
            is_active=lambda: True,
            go_stage=lambda *a: ui.navigate.to('/projects'),
            on_episode_change=_on_ep,
        )

    def _open_transfer():
        local = _sess.get('dir')
        if not local or not Path(local).is_dir():
            ui.notify('请先在上方选择一个 episode（用于确定 session 文件夹）',
                      type='warning')
            return
        dest_name = rsync_transfer.tagged_name(
            Path(local).name,
            app.storage.user.get('collector', ''),
            app.storage.user.get('username', ''))
        d = ui.dialog().props('persistent')
        with d, ui.card().style('width:760px; max-width:95vw; padding:24px; gap:12px;'):
            ui.label('传输本 session（rsync）').style('font-size:19px; font-weight:800;')
            ui.label(f'本地文件夹：{local}') \
                .style('font-size:12px; color:var(--text-3); word-break:break-all;')
            rs_host = ui.input('对方 (user@ip)', placeholder='yinan@10.99.120.4') \
                .props('outlined dense').style('width:100%;')
            rs_path = ui.input('对方文件位置（该 session 文件夹会以带姓名的新名字放进这里）',
                               placeholder='/run/media/.../bags/') \
                .props('outlined dense').style('width:100%;')
            dest_lbl = ui.label(f'传到：（对方文件位置）/{dest_name}') \
                .style('font-size:12px; color:var(--text-2); word-break:break-all;')
            rs_path.on_value_change(lambda e: dest_lbl.set_text(
                f'传到：{(e.value or "").strip().rstrip("/") or "（对方文件位置）"}'
                f'/{dest_name}'))
            rs_pass = ui.input('对方密码', password=True,
                               password_toggle_button=True) \
                .props('outlined dense').style('width:100%;')
            pct = ui.label('').style('font-size:14px;')
            bar = ui.linear_progress(value=0.0, show_value=False) \
                .props('rounded').style('width:100%;')
            log = ui.log(max_lines=300).style(
                'width:100%; height:200px; font-size:12px; font-family:monospace;')
            with ui.row().style('justify-content:flex-end; gap:8px; width:100%;'):
                ui.button('取消', on_click=d.close).props('flat color="grey-8"') \
                    .style('text-transform:none;')
                start_btn = ui.button('开始传输', icon='cloud_upload') \
                    .props('unelevated color="teal"') \
                    .style('text-transform:none; font-weight:600;')

            async def _go():
                host = (rs_host.value or '').strip()
                rpath = (rs_path.value or '').strip()
                if not (host and rpath):
                    ui.notify('请填写 对方 user@ip 和 文件位置', type='warning')
                    return
                miss = rsync_transfer.preflight()
                if miss:
                    ui.notify(f'本机缺少 {"、".join(miss)}，请先 apt install '
                              f'{" ".join(miss)}', type='negative', timeout=8000)
                    return
                start_btn.disable()
                pct.text = '0%'
                # 传文件夹内容到 <对方位置>/<带姓名的新文件夹名>/,本地文件夹名不变
                remote = f'{rpath.rstrip("/")}/{dest_name}/'
                log.push(f'→ {host}:{remote}')
                argv = rsync_transfer.build(local, host, remote, contents_only=True)
                try:
                    rc = await rsync_transfer.stream(
                        argv, rs_pass.value or '',
                        lambda p: (setattr(bar, 'value', p / 100.0),
                                   setattr(pct, 'text', f'{p}%')),
                        lambda ln: log.push(ln))
                except Exception as e:
                    rc = -1
                    log.push(f'错误: {e}')
                start_btn.enable()
                if rc == 0:
                    bar.value = 1.0
                    pct.text = '100% ✅ 传输完成'
                    ui.notify('rsync 传输完成', type='positive')
                else:
                    pct.text = f'❌ 失败 (rc={rc})'
                    ui.notify(f'rsync 失败(rc={rc}),看窗口日志', type='negative')
            start_btn.on('click', lambda: _go())
        d.open()
