#!/usr/bin/env python3
"""Full Annotate stage as a reusable builder.

Copied 2026-07-23 from app.py's inline Stage-5 block so the standalone Data
Studio GUI can host the EXACT same annotation UI without launching the teleop
SPA. `build_annotate(annotate_panel, *, pick_directory_dialog, is_active)`
renders the complete annotate UI (3-cam video + timeline + contact-mode C1–C6 +
subtask + typed annotators + episode nav/badges) into `annotate_panel`.

No ROS. Reads/writes per-episode json sidecars + mp4 via core.media mounts;
the caller must set S.active_project (for the typed-annotation submodules).
`pick_directory_dialog(start, title) -> path|None` backs the folder Browse;
`is_active() -> bool` gates the 1 Hz time tick + keyboard shortcuts.

NOTE: this is a copy of app.py's annotate block (teleop GUI left untouched on
purpose). Keep in sync if that block changes.
"""
from __future__ import annotations

import asyncio  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import subprocess  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
from collections import deque  # noqa: F401
from datetime import datetime  # noqa: F401
from pathlib import Path

from media import _media_url, _register_media_root  # noqa: F401
from nicegui import app, ui
from panels.project_annotate import build_project_annotate  # noqa: F401
from state import S


def build_annotate(annotate_panel, *, pick_directory_dialog=None,
                   is_active=None, go_stage=None, on_episode_change=None):
    """Build the full Annotate UI into `annotate_panel` (see module docstring).

    `on_episode_change(episode_dict)` — optional callback fired when the selected
    episode changes (dict has 'episode_dir'); the host uses it to offer
    "transfer current session".
    """
    _act = is_active if callable(is_active) else (lambda: True)
    go_stage = go_stage if callable(go_stage) else (lambda *a: ui.navigate.to('/app'))
    if pick_directory_dialog is None:
        async def pick_directory_dialog(start='~', title='Select folder'):
            return None
    # ─── ANNOTATE PAGE ────────────────────────────────────────
    # Contact-mode (C1-C6) segment annotation for converted LeRobot
    # episodes. Writes <episode>/contact_modes.json that the
    # converter (or downstream trainer) expands into per-frame labels.
    CONTACT_MODES = [
        ('C1', 'pre-shape',   '#5AC8FA'),
        ('C2', 'enclose',     '#34C759'),
        ('C3', 'pinch',       '#FF9500'),
        ('C4', 'reorient',    '#AF52DE'),
        ('C5', 'maintain',    '#FF2D55'),
        ('C6', 'release',     '#8E8E93'),
    ]
    _mode_color = {c[0]: c[2] for c in CONTACT_MODES}

    # Annotator state (per-session). Using lists so inner closures
    # can mutate without `nonlocal` gymnastics.
    annot_episodes: list[dict] = []   # [{'label','episode_dir','video_url'}, ...]
    annot_current = [None]            # selected episode dict
    annot_segments: list[list] = []   # each: [t_start_s, t_end_s, label]
    annot_duration = [0.0]            # seconds

    # User-toggle: hide failure episodes from the picker so the
    # operator can't accidentally burn time annotating an episode
    # that's already excluded from training. Default ON because the
    # subgoal pilot only trains on success eps. Toggle UI is built
    # below alongside the episode select.
    annot_hide_failures = [True]

    # Scan-folder ref — mirrors the Record page picker so operators
    # can point Annotate at an external SSD / sibling bag dir.
    # Shares the `session_browser_dir` storage key with Record so
    # changing one updates both; resolved fresh at panel build so
    # Record-side edits made earlier in the session take effect.
    _annot_root_ref = [Path(os.path.expanduser(
        app.storage.user.get('session_browser_dir', '~/teleop_bags')
    ))]
    # Register the initial root as a media mount up front so the
    # FIRST page render can serve mp4s under it. Idempotent.
    _register_media_root(_annot_root_ref[0])

    def _annotation_counts(ep_dir: Path) -> tuple[int, int]:
        """Return (contact_mode_segments, subtask_segments) counts
        from the per-episode sidecar files. Missing / malformed
        files return 0 for that category. Used to decorate the
        episode dropdown with status badges so the operator can
        see at a glance which episodes are done."""
        contact_n = 0
        sub_n = 0
        p = ep_dir / 'contact_modes.json'
        if p.is_file():
            try:
                contact_n = len(json.loads(
                    p.read_text(encoding='utf-8')
                ).get('segments', []))
            except Exception:
                pass
        p = ep_dir / 'subtasks.json'
        if p.is_file():
            try:
                sub_n = len(json.loads(
                    p.read_text(encoding='utf-8')
                ).get('segments', []))
            except Exception:
                pass
        return contact_n, sub_n

    def _project_annotation_counts(ep_dir: Path) -> list[tuple[str, int]]:
        """Per project-defined annotation field, count entries in its
        per-episode sidecar so the dropdown shows each field's progress:

          int/float/bool → <field>.json  → # segments
          detection      → bbox.json      → # annotated frames

        The subtask/action field is skipped (it IS the 🟢 badge).
        Plain string fields have no sidecar convention → skipped.
        """
        proj = getattr(S, 'active_project', None)
        specs = (proj or {}).get('annotations') or []
        out: list[tuple[str, int]] = []
        for a in specs:
            name = (a.get('name') or '').strip()
            typ = a.get('type') or 'string'
            if not name or (typ == 'string' and name in ('subtask', 'action')):
                continue
            n = 0
            if typ in ('int', 'float', 'bool'):
                p = ep_dir / f'{name}.json'
                if p.is_file():
                    try:
                        n = len(json.loads(p.read_text(encoding='utf-8'))
                                .get('segments', []))
                    except Exception:
                        pass
            elif typ == 'detection':
                p = ep_dir / 'bbox.json'
                if p.is_file():
                    try:
                        fr = json.loads(p.read_text(encoding='utf-8')).get('frames', {})
                        n = sum(1 for v in (fr or {}).values() if v)
                    except Exception:
                        pass
            else:
                continue  # plain string: no sidecar to count
            out.append((name, n))
        return out

    def _decorate_episode_label(ep_dir: Path, base: str) -> str:
        """Prepend annotation-progress badges to an episode label.

        🔴N = N contact-mode segments (contact_modes.json)
        🟢N = N subtask segments (subtasks.json)
        then one 🔵/⚪ <field>:N badge per project annotation field
        (🔵 = has data, ⚪ = none). All shown even when 0 so the
        operator can see at a glance which annotations are done.
        """
        c_n, s_n = _annotation_counts(ep_dir)
        extra = ''.join(f'  {"🔵" if n else "⚪"}{nm}:{n}'
                        for nm, n in _project_annotation_counts(ep_dir))
        return f'🔴{c_n}  🟢{s_n}{extra}   {base}'

    def _scan_lerobot_episodes() -> list[dict]:
        """Find converted episodes under <scan-folder>/*_lerobot/.

        An episode is a directory that has at least one mp4 under
        videos/ and a data.parquet or data.csv next to it.

        Filters out episodes whose meta.json reports
        outcome.success=False unless annot_hide_failures[0] is
        False (toggle in the page header).
        """
        root = _annot_root_ref[0]
        found: list[dict] = []
        if not root.exists():
            return found
        # When the scan folder IS a *_lerobot dir, its children are the
        # episodes → sort by NAME (episode_0000, 0001, …). Otherwise the
        # children are session dirs → newest-first by mtime. Either way
        # the dropdown AND the prev/next buttons follow this same order.
        if root.name.endswith('_lerobot'):
            _outer = sorted(root.iterdir(), key=lambda p: p.name)
        else:
            _outer = sorted(root.iterdir(),
                            key=lambda p: p.stat().st_mtime if p.exists() else 0,
                            reverse=True)
        for session_dir in _outer:
            if not session_dir.is_dir():
                continue
            # Either the session IS an episode (single-bag convert)
            # or it contains episode_* sub-dirs.
            candidates = [session_dir]
            # Sort episode_* by name so the list (and thus the dropdown
            # AND the prev/next buttons) is a stable numeric order
            # (episode_0000, 0001, …); iterdir() is unordered.
            candidates.extend(sorted(
                (p for p in session_dir.iterdir()
                 if p.is_dir() and p.name.startswith('episode_')),
                key=lambda p: p.name,
            ))
            for ep in candidates:
                vids = list((ep / 'videos').glob('*.mp4')) \
                        if (ep / 'videos').is_dir() else []
                if not vids:
                    continue
                # Outcome filter — read meta.json once per ep.
                # Treat missing / malformed meta as "unknown
                # outcome" → keep visible (better than silently
                # dropping legitimate data).
                if annot_hide_failures[0]:
                    ep_meta = ep / 'meta.json'
                    if ep_meta.is_file():
                        try:
                            m = json.loads(
                                ep_meta.read_text(encoding='utf-8'))
                            # `success` is written by two paths with
                            # different types: bool False (Preview) or
                            # str 'no'/'fail' (Log Trial Outcome form).
                            # Treat both as a failure to hide.
                            sv = m.get('outcome', {}).get('success')
                            is_fail = (sv is False) or (
                                isinstance(sv, str)
                                and sv.strip().lower() in ('no', 'fail', 'false'))
                            if is_fail:
                                continue
                        except Exception:
                            pass
                # Group by camera tag so the Annotate page can show
                # all three views in sync. Prefer the global view as
                # the master clock — it covers the workspace and is
                # the most informative for subtask boundaries.
                cam_to_path = {}
                for v in vids:
                    for cam in ('global', 'left_wrist', 'right_wrist'):
                        if cam in v.name and cam not in cam_to_path:
                            cam_to_path[cam] = v
                            break
                # Pick a master: global > left_wrist > first available.
                chosen = (cam_to_path.get('global')
                          or cam_to_path.get('left_wrist')
                          or vids[0])
                base_label = f'{session_dir.name} / {ep.name}' \
                    if ep != session_dir else session_dir.name
                label = _decorate_episode_label(ep, base_label)
                # Use _media_url so URLs pick up whichever media
                # mount actually covers `root` (default ~/teleop_bags
                # → /teleop-media/, picker-selected dirs →
                # /teleop-media-N/). Hardcoding /teleop-media/ here
                # would 404 for any non-default scan folder.
                # Cache-bust with the file's (mtime,size): the /teleop-
                # media-N prefix→dir mapping can change across restarts /
                # folder switches, so the SAME URL path may point at a
                # DIFFERENT file — without this the browser serves a
                # stale cached video (wrong episode). Distinct files →
                # distinct ?v= → fresh fetch; unchanged file → cacheable.
                def _bust(url: str, path: Path) -> str:
                    try:
                        st = path.stat()
                        return f'{url}?v={int(st.st_mtime)}_{st.st_size}'
                    except OSError:
                        return url
                chosen_url = _media_url(chosen)
                if not chosen_url:
                    # Mount missing — skip rather than emit a broken URL.
                    continue
                chosen_url = _bust(chosen_url, chosen)
                cam_urls = {}
                for cam, p in cam_to_path.items():
                    u = _media_url(p)
                    if u:
                        cam_urls[cam] = _bust(u, p)
                found.append({
                    'label': label,
                    'episode_dir': ep,
                    'video_path': chosen,
                    'video_url': chosen_url,
                    'cam_urls': cam_urls,  # {'global': url, 'left_wrist': url, ...}
                })
        return found

    def _load_existing_annotations(ep_dir: Path) -> tuple[list, float]:
        """Read contact_modes.json if present. Returns (segments, duration)."""
        p = ep_dir / 'contact_modes.json'
        if not p.is_file():
            return [], 0.0
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            segs = []
            for seg in data.get('segments', []):
                t_s = seg.get('t_start_ns', 0) / 1e9
                t_e = seg.get('t_end_ns', 0) / 1e9
                lbl = seg.get('label', '')
                segs.append([float(t_s), float(t_e), lbl])
            return segs, float(data.get('duration_s', 0.0))
        except Exception:
            return [], 0.0

    # Subtask annotation state — parallel to contact-mode segs but
    # carries free-text labels for long-horizon language conditioning.
    # Saved to subtasks.json sidecar; converter expands into LeRobot's
    # per-frame task_index column. Falls back to the recording-level
    # task when sidecar is absent (single-task data — no work needed).
    annot_subtasks: list[list] = []          # each: [t_start_s, t_end_s, text]
    annot_recent_subtasks: list[str] = []    # MRU dropdown source (cap 10)

    # ── Cube move timestamps (from cube_samples.json /cube/move) ──
    # Loaded automatically when an episode with cube_samples.json is selected.
    # Each entry: {'t_ns': int, 't_s': float, 'label': str}
    cube_moves: list[dict] = []
    # Predefined subtask labels for cube data
    CUBE_SUBTASK_LABELS = [
        'hold the cube in the starting grip',
        'rotate the whole cube',
        'turn the right layer clockwise',
        'turn the up layer clockwise',
        'turn the front layer clockwise',
        'turn the left layer clockwise',
        'turn the back layer clockwise',
        'turn the down layer clockwise',
    ]

    def _load_existing_subtasks(ep_dir: Path) -> list:
        """Read subtasks.json if present. Returns segments list."""
        p = ep_dir / 'subtasks.json'
        if not p.is_file():
            return []
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            segs = []
            for seg in data.get('segments', []):
                t_s = seg.get('t_start_ns', 0) / 1e9
                t_e = seg.get('t_end_ns', 0) / 1e9
                txt = seg.get('text', '')
                segs.append([float(t_s), float(t_e), str(txt)])
            return segs
        except Exception:
            return []

    def _load_cube_moves(ep_dir: Path) -> list[dict]:
        """Read cube_samples.json /cube/move timestamps.

        Returns list of {'t_ns': int, 't_s': float, 'label': str}
        sorted by time. Times are relative to the first quaternion
        sample (video start). If cube_samples.json is absent or has
        no /cube/move entries, returns [].
        """
        p = ep_dir / 'cube_samples.json'
        if not p.is_file():
            return []
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            moves = data.get('/cube/move', [])
            quats = data.get('/cube/quaternion', [])
            if not moves:
                return []
            t0 = quats[0]['timestamp_ns'] if quats else None
            # Read video start from parquet for accurate relative time
            if t0 is None:
                t0 = moves[0]['timestamp_ns']  # fallback: first move = t0
            parquet_path = ep_dir / 'data.parquet'
            if parquet_path.is_file():
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(str(parquet_path),
                                          columns=['timestamp_ns'])
                    t0 = table.column('timestamp_ns').to_pylist()[0]
                except ImportError:
                    pass  # pyarrow not available; use quat t0
                except Exception:
                    pass
            out = []
            for m in moves:
                t_ns = m['timestamp_ns'] - t0
                out.append({
                    't_ns': int(t_ns),
                    't_s': float(t_ns / 1e9),
                    'label': m.get('value', ''),
                })
            return sorted(out, key=lambda x: x['t_ns'])
        except Exception:
            return []

    with annotate_panel:
        ui.html('<div class="page-enter">')
        ui.html('''
            <div style="margin-bottom:20px;">
                <div style="font-size:32px; font-weight:800; letter-spacing:-0.5px; margin-bottom:6px;">Annotate</div>
                <div style="font-size:15px; opacity:0.4;">Contact-mode (C1–C6) + subtask (long-horizon language) labels. Writes <code>contact_modes.json</code> and <code>subtasks.json</code> per episode; converter bakes them into per-frame columns.</div>
            </div>
        ''')

        # Scan-folder picker — mirrors the Record page so the
        # operator can repoint Annotate at an external SSD or a
        # teammate's bag dir without restarting the GUI. Always
        # visible (above the empty-state gate) so it can be used
        # to recover from "No converted episodes yet". Shares the
        # `session_browser_dir` storage key with Record.
        with ui.element('div').classes('glass').style(
                'padding:14px 18px; margin-bottom:14px;'):
            with ui.row().style(
                    'width:100%; align-items:center; gap:8px;'):
                ui.html(
                    '<span style="font-size:12px; font-weight:500;'
                    ' min-width:90px; opacity:0.7;">'
                    'Scan folder</span>')
                annot_scan_input = ui.input(
                    value=str(_annot_root_ref[0])) \
                    .props('outlined dense') \
                    .style('flex:1; font-family:monospace;') \
                    .tooltip(
                        'Folder to scan for *_lerobot/ episodes. '
                        'Shared with the Record page; persisted '
                        'across GUI restarts.')

                def _apply_annot_scan_root(path_str: str):
                    new_root = Path(os.path.expanduser(
                        (path_str or '').strip() or '~/teleop_bags'))
                    _annot_root_ref[0] = new_root
                    app.storage.user['session_browser_dir'] = str(
                        new_root)
                    annot_scan_input.set_value(str(new_root))
                    # Register the new root as a media mount so the
                    # browser can fetch mp4 / parquet under it.
                    # _register_media_root is idempotent — calling
                    # it for ~/teleop_bags (already mounted at
                    # startup) returns the existing prefix.
                    _register_media_root(new_root)
                    _rescan_and_repop()

                async def _on_browse_annot_scan_root():
                    picked = await pick_directory_dialog(
                        str(_annot_root_ref[0]),
                        'Select scan folder')
                    if picked == 'NO_ZENITY':
                        ui.notify(
                            'zenity not installed — type the path '
                            'manually then click Apply',
                            type='warning', position='bottom',
                            timeout=2800)
                        return
                    if not picked:
                        return
                    _apply_annot_scan_root(picked)
                    ui.notify(f'Scanning {picked}',
                              type='positive', position='bottom',
                              timeout=1800)

                ui.button('Browse…', icon='folder_open') \
                    .props('outline color="primary" size="sm"') \
                    .style(
                        'border-radius:8px; text-transform:none; '
                        'font-weight:600; font-size:12px;') \
                    .on('click', _on_browse_annot_scan_root)
                ui.button('Apply', icon='check') \
                    .props('outline color="primary" size="sm"') \
                    .style(
                        'border-radius:8px; text-transform:none; '
                        'font-weight:600; font-size:12px;') \
                    .on('click', lambda: _apply_annot_scan_root(
                        annot_scan_input.value)) \
                    .tooltip('Apply typed path')
                ui.button('Reset', icon='restart_alt') \
                    .props('flat color="grey" size="sm"') \
                    .style(
                        'text-transform:none; font-weight:500; '
                        'font-size:12px;') \
                    .on('click', lambda: _apply_annot_scan_root(
                        '~/teleop_bags')) \
                    .tooltip('Reset to ~/teleop_bags')

        # Empty-state gate: shown when no converted episodes exist.
        annot_gate_panel = ui.element('div').style('display:none;')
        with annot_gate_panel:
            with ui.element('div').classes('glass').style(
                'padding:48px 32px; margin-bottom:16px; display:flex; '
                'flex-direction:column; align-items:center; gap:14px; text-align:center;'
            ):
                ui.icon('movie_filter', size='52px').style('opacity:0.2;')
                ui.html('<div style="font-size:18px; font-weight:700; opacity:0.6;">No converted episodes yet</div>')
                ui.html(
                    '<div style="font-size:13px; opacity:0.45; max-width:420px; line-height:1.6;">'
                    'Annotate works on LeRobot-format output (mp4 + data.parquet). '
                    'Record a session, then press <b>Convert → LeRobot</b> on the Record page '
                    'to unlock this page.</div>'
                )
                with ui.row().style('gap:10px; margin-top:8px;'):
                    ui.button('Go to Record page', icon='fiber_manual_record') \
                        .props('unelevated color="primary"') \
                        .style('border-radius:12px; text-transform:none; font-weight:600; padding:10px 22px;') \
                        .on('click', lambda: go_stage(4))
                    ui.button('Rescan', icon='refresh') \
                        .props('outline color="grey"') \
                        .style('border-radius:12px; text-transform:none; font-weight:600;') \
                        .on('click', lambda: _rescan_and_repop())

        annot_work_panel = ui.element('div')
        with annot_work_panel:
            with ui.element('div').classes('glass').style('padding:20px; margin-bottom:14px;'):
                with ui.row().style('width:100%; align-items:center; gap:14px;'):
                    ui.html('<span style="font-size:13px; font-weight:600; opacity:0.7; min-width:80px;">Episode</span>')
                    annot_select = ui.select([], value=None) \
                        .props('dense outlined options-dense') \
                        .style('flex:1; min-width:320px; font-family:monospace; font-size:12px;')
                    ui.button('Refresh', icon='refresh') \
                        .props('outline color="grey" size="sm"') \
                        .style('border-radius:10px; text-transform:none; font-weight:600; font-size:12px;') \
                        .on('click', lambda: _rescan_and_repop())

                    # Outcome filter — defaults to hiding failures
                    # (they're excluded from training anyway).
                    def _toggle_hide_failures(e):
                        annot_hide_failures[0] = bool(e.value)
                        _rescan_and_repop()
                    ui.checkbox(
                        'Hide failures', value=True,
                        on_change=_toggle_hide_failures,
                    ).props('dense color="primary"') \
                        .style('font-size:12px;') \
                        .tooltip(
                            'Hide episodes whose outcome was marked '
                            'failed before conversion. They are not '
                            'in the training set, so labeling them '
                            'is wasted effort.')

            # Video + timeline + labels panel
            with ui.element('div').classes('glass').style('padding:20px; margin-bottom:14px;'):
                # Three-up camera view: global is the master (full
                # controls + currentTime source), wrist views are
                # JS-synced read-only mirrors. CSS class hooks let
                # the sync script find them across episode swaps.
                with ui.row().style(
                        'gap:8px; width:100%; align-items:flex-start; '
                        'flex-wrap:nowrap;'):
                    with ui.element('div').style(
                            'flex:1; min-width:0; display:flex; '
                            'flex-direction:column; gap:4px;'):
                        ui.html('<div style="font-size:10px; '
                                'font-weight:700; opacity:0.55; '
                                'letter-spacing:0.6px;">'
                                'GLOBAL · MASTER</div>')
                        annot_video = (ui.video('').props('controls')
                            .classes('v4-annot-video '
                                      'v4-annot-master')
                            .style('width:100%; max-height:280px; '
                                   'background:#000; '
                                   'border-radius:8px;'))
                    with ui.element('div').style(
                            'flex:1; min-width:0; display:flex; '
                            'flex-direction:column; gap:4px;'):
                        ui.html('<div style="font-size:10px; '
                                'font-weight:700; opacity:0.55; '
                                'letter-spacing:0.6px;">'
                                'LEFT WRIST</div>')
                        annot_video_lwrist = (ui.video('')
                            .props('muted')
                            .classes('v4-annot-video '
                                      'v4-annot-slave')
                            .style('width:100%; max-height:280px; '
                                   'background:#000; '
                                   'border-radius:8px; '
                                   'pointer-events:none;'))
                    with ui.element('div').style(
                            'flex:1; min-width:0; display:flex; '
                            'flex-direction:column; gap:4px;'):
                        ui.html('<div style="font-size:10px; '
                                'font-weight:700; opacity:0.55; '
                                'letter-spacing:0.6px;">'
                                'RIGHT WRIST</div>')
                        annot_video_rwrist = (ui.video('')
                            .props('muted')
                            .classes('v4-annot-video '
                                      'v4-annot-slave')
                            .style('width:100%; max-height:280px; '
                                   'background:#000; '
                                   'border-radius:8px; '
                                   'pointer-events:none;'))

                # Timeline (segments as colored blocks)
                ui.html('<div style="font-size:11px; font-weight:700; opacity:0.5; letter-spacing:0.5px; margin:14px 0 6px;">TIMELINE</div>')
                annot_timeline_el = ui.html('').style(
                    'width:100%; height:28px; background:rgba(0,0,0,0.06); '
                    'border-radius:6px; overflow:hidden; position:relative;')
                annot_time_label = ui.html('<div style="font-size:11px; opacity:0.4; '
                                            'font-family:monospace; margin-top:4px;">0.00s / 0.00s</div>')

                # ── Project-spec annotation sub-modules (own module, §13) ──
                # Horizontal nav below the video, one entry per annotation
                # type. detection-type tabs host the bbox annotator (§14);
                # ctx gives it the current episode + master-video selector,
                # and collects per-tab reload() fns to run on episode change.
                # Returns per-field tab panels (subtask UI moves into "action").
                _bbox_reloaders: list = []
                _ann_panels = build_project_annotate({
                    'get_episode_dir': lambda: (annot_current[0] or {}).get('episode_dir'),
                    'video_selector': 'video.v4-annot-master',
                    'reloaders': _bbox_reloaders,
                    # late-bound: refresh dropdown badges after a project
                    # annotation (int/float/bool/detection) is saved.
                    'on_saved': lambda: _refresh_badges_keep_selection(),
                })

                # ── Prev / Next episode quick-switch (below the
                # annotation sub-modules). `_go_episode` is defined
                # later; resolved at click time (late binding). ──
                with ui.row().style('gap:10px; align-items:center; '
                                    'margin:16px 0 6px; flex-wrap:wrap;'):
                    ui.button('上一条 episode', icon='chevron_left',
                              on_click=lambda: _go_episode(-1)) \
                        .props('outline color="primary"') \
                        .style('text-transform:none;')
                    ui.button('下一条 episode',
                              on_click=lambda: _go_episode(1)) \
                        .props('unelevated color="primary" icon-right=chevron_right') \
                        .style('text-transform:none;')
                    ep_nav_pos = ui.label('') \
                        .style('font-size:12px; color:var(--text-3);')

                # Label buttons
                ui.html('<div style="font-size:11px; font-weight:700; opacity:0.5; letter-spacing:0.5px; margin:14px 0 6px;">LABEL AT CURRENT TIME</div>')
                with ui.row().style('gap:6px; flex-wrap:wrap;'):
                    def _make_label_click(lbl: str):
                        return lambda: _split_at_current_time(lbl)
                    for code, name, color in CONTACT_MODES:
                        ui.button(f'{code}  {name}') \
                            .props('unelevated') \
                            .style(f'background:{color}; color:white; border-radius:8px; '
                                   f'text-transform:none; font-weight:600; font-size:12px; '
                                   f'padding:6px 14px; min-width:0;') \
                            .on('click', _make_label_click(code))
                    ui.button('Undo last', icon='undo') \
                        .props('outline color="grey"') \
                        .style('border-radius:8px; text-transform:none; font-weight:600; font-size:12px;') \
                        .on('click', lambda: _undo_last_segment())
                    ui.button('Clear all', icon='clear') \
                        .props('outline color="negative"') \
                        .style('border-radius:8px; text-transform:none; font-weight:600; font-size:12px;') \
                        .on('click', lambda: _clear_segments())

                # Save / status
                with ui.row().style('margin-top:14px; gap:10px; align-items:center;'):
                    ui.button('Save contact_modes.json', icon='save') \
                        .props('unelevated color="primary"') \
                        .style('border-radius:10px; text-transform:none; font-weight:600; font-size:13px;') \
                        .on('click', lambda: _save_annotations())
                    annot_status = ui.html('<span style="font-size:12px; opacity:0.5;">Pick an episode to begin</span>')

                # ── Subtask (long-horizon language) UI [subtask-in-tab] ──
                # Built into a box, then MOVED into the matching annotation
                # sub-module tab (string field, e.g. "action"). If there is
                # no such field, it stays here (feature preserved).
                _subtask_box = ui.element('div').style('width:100%;')
                with _subtask_box:
                    # ── Subtask track (long-horizon language labels) ──
                    ui.html(
                        '<div style="height:1px; background:rgba(0,0,0,0.08); '
                        'margin:22px 0 14px;"></div>')
                    ui.html(
                        '<div style="font-size:11px; font-weight:700; opacity:0.5; '
                        'letter-spacing:0.5px; margin:0 0 6px;">'
                        'SUBTASK · LONG-HORIZON LANGUAGE LABEL</div>')
                    sub_timeline_el = ui.html('').style(
                        'width:100%; height:24px; background:rgba(0,0,0,0.06); '
                        'border-radius:6px; overflow:hidden; position:relative; '
                        'margin-bottom:10px;')
                    # Tabular view of the segments — start / end / dur.
                    # Updated alongside the colored bar by
                    # `_render_subtask_timeline()`. Empty by default.
                    sub_list_el = ui.html('').style('margin:0 0 12px 0;')
                    with ui.row().style(
                            'gap:8px; align-items:center; flex-wrap:wrap;'):
                        sub_input = ui.input(
                            placeholder='e.g. pick the plush toy / open drawer'
                        ).props('dense outlined clearable').classes(
                            'v4-subtask-input').style(
                            'flex:0 1 460px; min-width:220px; max-width:520px;')
                        sub_recent_select = ui.select(
                            [], value=None, label='Recent') \
                            .props('dense outlined options-dense clearable') \
                            .style('min-width:180px; font-size:12px;')
                        ui.button('Add at cursor', icon='add') \
                            .props('unelevated color="primary" size="sm"') \
                            .style('border-radius:8px; text-transform:none; '
                                   'font-weight:700; font-size:12px;') \
                            .on('click', lambda:
                                _split_subtask_at_current_time())
                    with ui.row().style(
                            'gap:8px; margin-top:10px; align-items:center; '
                            'flex-wrap:wrap;'):
                        ui.button('Undo last subtask', icon='undo') \
                            .props('outline color="grey" size="sm"') \
                            .style('border-radius:8px; text-transform:none; '
                                   'font-weight:600; font-size:12px;') \
                            .on('click', lambda: _undo_last_subtask())
                        ui.button('Clear all subtasks', icon='clear') \
                            .props('outline color="negative" size="sm"') \
                            .style('border-radius:8px; text-transform:none; '
                                   'font-weight:600; font-size:12px;') \
                            .on('click', lambda: _clear_subtasks())
                        ui.button('Save subtasks.json', icon='save') \
                            .props('unelevated color="primary" size="sm"') \
                            .style('border-radius:8px; text-transform:none; '
                                   'font-weight:700; font-size:12px;') \
                            .on('click', lambda: _save_subtasks())
                        sub_status = ui.html(
                            '<span style="font-size:12px; opacity:0.5;">'
                            'No subtasks yet</span>')

                    # ── Cube Move Markers (from cube_samples.json) ──
                    # Auto-loaded when episode has cube_samples.json with
                    # /cube/move entries. Shows move timestamps as markers
                    # on the timeline; user assigns subtask labels to each
                    # segment between consecutive moves.
                    cube_move_panel = ui.element('div').style(
                        'display:none; width:100%;')
                    with cube_move_panel:
                        ui.html(
                            '<div style="height:1px; background:rgba(0,0,0,0.08); '
                            'margin:22px 0 14px;"></div>')
                        ui.html(
                            '<div style="font-size:11px; font-weight:700; opacity:0.5; '
                            'letter-spacing:0.5px; margin:0 0 6px;">'
                            'CUBE MOVES · AUTO-DETECTED FROM cube_samples.json</div>')

                        # Timeline with move markers
                        cube_timeline_el = ui.html('').style(
                            'width:100%; height:28px; background:rgba(0,0,0,0.06); '
                            'border-radius:6px; overflow:hidden; position:relative; '
                            'margin-bottom:10px;')

                        # Move markers list + label selectors
                        cube_moves_list_el = ui.html('').style(
                            'margin:0 0 12px 0;')

                        with ui.row().style(
                                'gap:8px; margin-top:10px; align-items:center; '
                                'flex-wrap:wrap;'):
                            ui.button('Generate subtasks from moves',
                                      icon='auto_fix_high') \
                                .props('unelevated color="teal" size="sm') \
                                .style('border-radius:8px; text-transform:none; '
                                       'font-weight:700; font-size:12px;') \
                                .on('click', lambda:
                                    _generate_subtasks_from_moves())
                            ui.button('Clear cube moves', icon='clear') \
                                .props('outline color="negative" size="sm"') \
                                .style('border-radius:8px; text-transform:none; '
                                       'font-weight:600; font-size:12px;') \
                                .on('click', lambda: _clear_cube_moves())
                            cube_status = ui.html(
                                '<span style="font-size:12px; opacity:0.5;">'
                                'No cube moves loaded</span>')

                    # 数字键 1–0 速填 subtask 预设(临时 · 退出本页即清空)
                    preset_inputs = []
                    with ui.expansion(
                            '数字键 1–0 速填 subtask（临时 · 退出本页清空）',
                            icon='keyboard', value=True).classes('w-full') \
                            .style('margin-top:12px;'):
                        ui.label('给 1–9、0 各设一个常用 subtask;标注时（焦点不在'
                                 '输入框）按对应数字键 = 在光标处添加该 subtask。') \
                            .style('font-size:11px; color:var(--text-3); '
                                   'margin-bottom:8px;')
                        with ui.row().style('gap:6px 18px; width:100%; '
                                            'flex-wrap:wrap;'):
                            for _dg in ['1', '2', '3', '4', '5', '6',
                                        '7', '8', '9', '0']:
                                with ui.row().style(
                                        'gap:8px; align-items:center; '
                                        'flex:1 1 300px; min-width:260px; '
                                        'flex-wrap:nowrap;'):
                                    ui.label(_dg).style(
                                        'font-weight:800; font-size:14px; '
                                        'width:18px; text-align:center; '
                                        'color:var(--blue); flex-shrink:0;')
                                    preset_inputs.append(
                                        ui.input(placeholder=f'键 {_dg} 的 subtask')
                                        .props('dense outlined clearable')
                                        .style('flex:1; min-width:0;'))

                    ui.html('''
                        <div style="font-size:11px; opacity:0.5; margin-top:14px; line-height:1.6;">
                            Keyboard: <b>Space</b> play/pause · <b>←/→</b> seek ±0.5s ·
                            <b>Shift+←/→</b> ±0.1s · <b>Enter</b>(输入框内) add at cursor ·
                            <b>1–9,0</b>(焦点不在输入框时) 速填对应预设 subtask
                            <br>接触模式 C1–C6 请用下方按钮点击。Single-task 数据可不标 subtask，
                            converter 会回退到 recording-level task。
                        </div>
                    ''')

                # [subtask-in-tab] host the subtask box inside the matching
                # annotation sub-module tab: prefer a field named "action",
                # else the first string-type field. If none, it stays here.
                _sub_host = _ann_panels.get('action')
                if _sub_host is None:
                    for _a in ((S.active_project or {}).get('annotations', []) or []):
                        if _a.get('type') == 'string':
                            _sub_host = _ann_panels.get(_a.get('name'))
                            break
                if _sub_host is not None:
                    _sub_host.clear()
                    _subtask_box.move(_sub_host)
                # (独立标注工具:已移除"Re-bake into training dataset"按钮——
                #  它依赖 ROS 转换器 scripts/rosbag2_to_lerobot.py,不在本项目内。)

        ui.html('</div>')

        # Master/slave video sync — edge-triggered. Earlier version
        # synced on `timeupdate` (fires 30-60 Hz), which during a
        # scrub forces each slave to re-seek + re-decode every
        # frame the user drags through → laggy. Now we sync only
        # on play/pause/seeked + a low-frequency drift check (2 Hz)
        # while the master is actively playing.
        ui.add_body_html('''
            <script>
            (function() {
              const SEL_MASTER = 'video.v4-annot-master, .v4-annot-master video';
              const SEL_SLAVE  = 'video.v4-annot-slave, .v4-annot-slave video';
              function master() { return document.querySelector(SEL_MASTER); }
              function slaves() { return document.querySelectorAll(SEL_SLAVE); }
              function applyTime(t) {
                  slaves().forEach(s => {
                      try { s.currentTime = t; } catch (e) {}
                  });
              }
              function attach() {
                const m = master();
                if (!m || m.dataset.v4SyncWired === '1') return;
                m.dataset.v4SyncWired = '1';
                m.addEventListener('play', () => slaves().forEach(
                    s => { s.play().catch(()=>{}); }));
                m.addEventListener('pause', () => slaves().forEach(
                    s => s.pause()));
                m.addEventListener('seeking', () => slaves().forEach(
                    s => s.pause()));  // freeze slaves while user drags
                m.addEventListener('seeked', () => {
                    applyTime(m.currentTime);
                    if (!m.paused) {
                        slaves().forEach(s => { s.play().catch(()=>{}); });
                    }
                });
                m.addEventListener('ratechange', () => slaves().forEach(
                    s => { s.playbackRate = m.playbackRate; }));
              }
              // 数字键 1–0 = subtask 预设速填。用 capture 阶段的文档监听,
              // 抢在 <video> 原生"数字跳转 0–90%"之前 preventDefault,这样
              // 即使鼠标焦点在视频/视频轴上,数字键也照样触发标注。
              // 只有焦点在真正的文本输入框里时才放行(让你正常打数字)。
              document.addEventListener('keydown', function (ev) {
                  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
                  if (!/^[0-9]$/.test(ev.key)) return;
                  var el = document.activeElement;
                  var tag = (el && el.tagName || '').toLowerCase();
                  if (tag === 'input' || tag === 'textarea' || tag === 'select'
                      || (el && el.isContentEditable)) return;
                  ev.preventDefault();
                  ev.stopPropagation();
                  if (typeof emitEvent === 'function') emitEvent('subtask_preset', ev.key);
              }, true);
              // Re-check master every 500ms so episode swaps re-arm.
              setInterval(attach, 500);
              // Slow drift correction during playback (only fires
              // when actually playing AND not currently seeking).
              setInterval(() => {
                  const m = master();
                  if (!m || m.paused || m.seeking) return;
                  slaves().forEach(s => {
                      if (Math.abs(s.currentTime - m.currentTime) > 0.5) {
                          try { s.currentTime = m.currentTime; } catch (e) {}
                      }
                  });
              }, 500);
              // Cube move seek/label/delete click delegation
              document.addEventListener('click', function(ev) {
                  var el = ev.target;
                  if (!el) return;
                  if (el.classList.contains('cube-seek')) {
                      var t = parseFloat(el.getAttribute('data-time') || '0');
                      var v = document.querySelector('video.v4-annot-master, video');
                      if (v && !isNaN(t)) {
                          v.currentTime = t;
                          v.pause();
                      }
                  } else if (el.classList.contains('cube-move-label-select')) {
                      var seg = el.getAttribute('data-seg');
                      if (seg !== null && typeof emitEvent === 'function')
                          emitEvent('cube_move_label', {seg: parseInt(seg)});
                  } else if (el.classList.contains('cube-move-delete')) {
                      var seg = el.getAttribute('data-seg');
                      if (seg !== null && typeof emitEvent === 'function')
                          emitEvent('cube_move_delete', {seg: parseInt(seg)});
                  }
              });
            })();
            </script>
        ''')

    # ── Annotate page logic (attached after panel built) ──

    def _render_timeline():
        d = annot_duration[0]
        if d <= 0:
            annot_timeline_el.set_content('')
            return
        parts = []
        last_end = 0.0
        for seg in annot_segments:
            t0, t1, lbl = seg
            if t0 > last_end:
                # gap
                w = (t0 - last_end) / d * 100
                parts.append(f'<div style="position:absolute; left:{last_end/d*100:.2f}%; '
                              f'top:0; bottom:0; width:{w:.2f}%; background:transparent;"></div>')
            w = max(0.2, (t1 - t0) / d * 100)
            color = _mode_color.get(lbl, '#999')
            parts.append(
                f'<div title="{lbl}: {t0:.2f}s → {t1:.2f}s" '
                f'style="position:absolute; left:{t0/d*100:.2f}%; top:0; bottom:0; '
                f'width:{w:.2f}%; background:{color}; display:flex; align-items:center; '
                f'justify-content:center; color:white; font-size:11px; font-weight:700; '
                f'border-right:1px solid rgba(255,255,255,0.3);">{lbl}</div>'
            )
            last_end = t1
        annot_timeline_el.set_content(''.join(parts))

    def _get_video_time_sync() -> float:
        """Pull current video time (s) from the browser via JS round-trip."""
        try:
            return float(ui.run_javascript(
                'const v = document.querySelector("video.v4-annot-master, video");'
                'v ? v.currentTime : 0',
                timeout=0.5) or 0.0)
        except Exception:
            return 0.0

    async def _get_video_time_async() -> float:
        try:
            t = await ui.run_javascript(
                'const v = document.querySelector("video.v4-annot-master, video");'
                'v ? v.currentTime : 0', timeout=0.5)
            return float(t or 0.0)
        except Exception:
            return 0.0

    def _split_at_current_time(label: str):
        if annot_current[0] is None:
            ui.notify('Pick an episode first', type='warning',
                      position='bottom', timeout=1500)
            return

        async def _do():
            t = await _get_video_time_async()
            if t <= 0:
                ui.notify('Video not ready', type='warning',
                          position='bottom', timeout=1500)
                return
            # Close the last open segment (if any) at t, open a new one.
            if annot_segments:
                last = annot_segments[-1]
                if t > last[1]:  # only if cursor advanced past last end
                    annot_segments.append([last[1], t, last[2]])  # fill gap with same label
                last[1] = t
            # The NEW segment is [t, duration, label] (end-anchored to duration)
            annot_segments.append([t, annot_duration[0], label])
            _render_timeline()
            annot_status.set_content(
                f'<span style="font-size:12px; color:var(--green);">'
                f'Marked {label} at {t:.2f}s · {len(annot_segments)} segs</span>')
        ui.timer(0.01, _do, once=True)

    def _undo_last_segment():
        if not annot_segments:
            return
        annot_segments.pop()
        _render_timeline()
        annot_status.set_content(
            f'<span style="font-size:12px;">Undone · {len(annot_segments)} segs</span>')

    def _clear_segments():
        annot_segments.clear()
        _render_timeline()
        annot_status.set_content(
            '<span style="font-size:12px;">All segments cleared</span>')

    def _save_annotations():
        cur = annot_current[0]
        if cur is None:
            ui.notify('Pick an episode first', type='warning',
                      position='bottom', timeout=1500)
            return
        if not annot_segments:
            ui.notify('No segments to save', type='warning',
                      position='bottom', timeout=1500)
            return
        payload = {
            'version': 1,
            'schema': 'C1:pre-shape, C2:enclose, C3:pinch, C4:reorient, C5:maintain, C6:release',
            'duration_s': float(annot_duration[0]),
            'segments': [
                {'t_start_ns': int(s[0] * 1e9),
                 't_end_ns':   int(s[1] * 1e9),
                 'label': s[2]}
                for s in annot_segments
            ],
            'annotator': app.storage.user.get('username', 'unknown'),
            'annotated_at': datetime.now().isoformat(timespec='seconds'),
        }
        try:
            out_path = cur['episode_dir'] / 'contact_modes.json'
            tmp = out_path.with_suffix(out_path.suffix + '.tmp')
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding='utf-8')
            os.replace(tmp, out_path)
            annot_status.set_content(
                f'<span style="font-size:12px; color:var(--green); font-weight:600;">'
                f'Saved {len(annot_segments)} segs → {out_path.name}</span>')
            ui.notify(f'Saved {out_path}', type='positive',
                      position='bottom', timeout=2200)
            # Refresh dropdown badges so 🔴N updates for the
            # just-saved episode without manual Refresh.
            _refresh_badges_keep_selection()
        except Exception as e:
            ui.notify(f'Save failed: {e}', type='negative',
                      position='bottom', timeout=3000)

    # ── Subtask annotation logic (parallel to contact-mode) ─────

    def _color_for_subtask(text: str) -> str:
        """Stable hash → palette index, so the same text reuses the
        same color across episodes."""
        palette = ['#5AC8FA', '#34C759', '#FF9500', '#AF52DE',
                   '#FF2D55', '#FFCC00', '#5856D6', '#00C7BE']
        h = sum(ord(c) for c in text) % len(palette)
        return palette[h]

    def _render_subtask_timeline():
        d = annot_duration[0]
        if d <= 0:
            sub_timeline_el.set_content('')
            sub_list_el.set_content('')
            return
        parts = []
        for seg in annot_subtasks:
            t0, t1, txt = seg
            # Resolve open final segment (t_end <= t_start, e.g.
            # loaded from an old sidecar with t_end_ns=0) to the
            # episode duration so the bar actually renders.
            t1_eff = t1 if t1 > t0 else d
            w = max(0.3, (t1_eff - t0) / d * 100)
            color = _color_for_subtask(txt)
            safe = (txt.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;'))
            parts.append(
                f'<div title="{safe}: {t0:.2f}s → {t1_eff:.2f}s" '
                f'style="position:absolute; left:{t0/d*100:.2f}%; '
                f'top:0; bottom:0; width:{w:.2f}%; background:{color}; '
                f'display:flex; align-items:center; '
                f'justify-content:center; color:white; font-size:11px; '
                f'font-weight:600; border-right:1px solid '
                f'rgba(255,255,255,0.3); padding:0 6px; '
                f'overflow:hidden; white-space:nowrap; '
                f'text-overflow:ellipsis;">{safe}</div>')
        sub_timeline_el.set_content(''.join(parts))

        # ── Tabular list below the bar ──
        # Lets the operator read off start / end / duration per
        # segment without hovering tooltips. Resolves t_end=0
        # (last segment still open at save-time) to the episode
        # duration so the duration column never reads negative.
        if not annot_subtasks:
            sub_list_el.set_content(
                '<div style="font-size:11px; opacity:0.4; '
                'font-style:italic;">No subtask segments yet — '
                'add at cursor to start labeling.</div>')
            return
        rows = [
            '<table style="width:100%; border-collapse:collapse; '
            'font-size:12px; font-family:monospace;">'
            '<thead><tr style="text-align:left; opacity:0.55; '
            'font-weight:700; letter-spacing:0.5px;">'
            '<th style="padding:4px 8px;">#</th>'
            '<th style="padding:4px 8px;">SUBGOAL</th>'
            '<th style="padding:4px 8px; text-align:right;">START</th>'
            '<th style="padding:4px 8px; text-align:right;">END</th>'
            '<th style="padding:4px 8px; text-align:right;">DUR</th>'
            '<th style="padding:4px 8px; text-align:right;">%</th>'
            '</tr></thead><tbody>'
        ]
        total = sum(
            max(0.0, (s[1] if s[1] > s[0] else d) - s[0])
            for s in annot_subtasks)
        _OPEN_TAG = (
            ' <span style="opacity:0.4;">(open)</span>')
        for i, seg in enumerate(annot_subtasks):
            t0, t1, txt = seg
            # Resolve open final segment to duration `d`.
            t1_eff = t1 if t1 > t0 else d
            dur = max(0.0, t1_eff - t0)
            pct = (dur / total * 100) if total > 0 else 0.0
            color = _color_for_subtask(txt)
            safe = (txt.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;'))
            open_marker = '' if t1 > t0 else _OPEN_TAG
            rows.append(
                f'<tr style="border-top:1px solid '
                f'rgba(0,0,0,0.06);">'
                f'<td style="padding:4px 8px; opacity:0.5;">'
                f'{i + 1}</td>'
                f'<td style="padding:4px 8px;">'
                f'<span style="display:inline-block; width:8px; '
                f'height:8px; border-radius:2px; '
                f'background:{color}; margin-right:8px; '
                f'vertical-align:middle;"></span>{safe}</td>'
                f'<td style="padding:4px 8px; text-align:right; '
                f'opacity:0.75;">{t0:6.2f}s</td>'
                f'<td style="padding:4px 8px; text-align:right; '
                f'opacity:0.75;">{t1_eff:6.2f}s{open_marker}</td>'
                f'<td style="padding:4px 8px; text-align:right; '
                f'font-weight:700;">{dur:5.2f}s</td>'
                f'<td style="padding:4px 8px; text-align:right; '
                f'opacity:0.55;">{pct:4.1f}%</td>'
                f'</tr>')
        rows.append(
            f'<tr style="border-top:2px solid '
            f'rgba(0,0,0,0.12); font-weight:700;">'
            f'<td colspan="4" style="padding:6px 8px; '
            f'text-align:right; opacity:0.6;">'
            f'{len(annot_subtasks)} segs · '
            f'video {d:.2f}s</td>'
            f'<td style="padding:6px 8px; text-align:right;">'
            f'{total:5.2f}s</td>'
            f'<td style="padding:6px 8px; text-align:right; '
            f'opacity:0.55;">'
            f'{(total / d * 100) if d > 0 else 0:4.1f}%</td>'
            f'</tr>')
        rows.append('</tbody></table>')
        sub_list_el.set_content(''.join(rows))

    def _refresh_recent_select():
        sub_recent_select.options = list(annot_recent_subtasks)
        try:
            sub_recent_select.update()
        except Exception:
            pass

    def _push_recent(text: str):
        # MRU: most recent first; cap at 10.
        if not text:
            return
        if text in annot_recent_subtasks:
            annot_recent_subtasks.remove(text)
        annot_recent_subtasks.insert(0, text)
        del annot_recent_subtasks[10:]
        _refresh_recent_select()

    def _split_subtask_at_current_time(text=None):
        if annot_current[0] is None:
            ui.notify('Pick an episode first', type='warning',
                      position='bottom', timeout=1500)
            return
        text = (text if text is not None else (sub_input.value or '')).strip()
        if not text:
            ui.notify('Type a subtask first', type='warning',
                      position='bottom', timeout=1500)
            return

        async def _do():
            t = await _get_video_time_async()
            # Close last open segment at t (if cursor advanced past it),
            # then append a new segment [t, duration, text].
            if annot_subtasks:
                last = annot_subtasks[-1]
                if t <= last[0]:
                    ui.notify('Cursor is at or before last segment '
                              'start — seek forward first',
                              type='warning',
                              position='bottom', timeout=1800)
                    return
                last[1] = t  # close gap to cursor
            annot_subtasks.append([float(t), float(annot_duration[0]),
                                   text])
            _push_recent(text)
            _render_subtask_timeline()
            sub_status.set_content(
                f'<span style="font-size:12px; color:var(--green);">'
                f'Added "{text}" at {t:.2f}s · '
                f'{len(annot_subtasks)} segs</span>')
        ui.timer(0.01, _do, once=True)

    def _apply_preset_subtask(idx: int):
        """数字键速填:取第 idx 个预设文本,在光标处添加为 subtask 分段。
        预设只存内存(退出本页即清空)。"""
        try:
            txt = (preset_inputs[idx].value or '').strip()
        except (IndexError, AttributeError, NameError):
            txt = ''
        if not txt:
            ui.notify('该数字键还没设置 subtask（在上方「数字键速填」里填）',
                      type='warning', position='bottom', timeout=1400)
            return
        _split_subtask_at_current_time(text=txt)

    def _undo_last_subtask():
        if not annot_subtasks:
            return
        annot_subtasks.pop()
        _render_subtask_timeline()
        sub_status.set_content(
            f'<span style="font-size:12px;">Undone · '
            f'{len(annot_subtasks)} segs</span>')

    def _clear_subtasks():
        annot_subtasks.clear()
        _render_subtask_timeline()
        sub_status.set_content(
            '<span style="font-size:12px;">All subtasks cleared</span>')

    # ── Cube move logic ──────────────────────────────────────────

    def _render_cube_moves():
        """Render the cube move timeline bar and the move list with
        label selectors."""
        moves = cube_moves
        dur = annot_duration[0]
        if not moves or dur <= 0:
            cube_timeline_el.set_content('')
            cube_moves_list_el.set_content('')
            cube_move_panel.style('display:none;')
            return

        cube_move_panel.style('display:block;')

        # ── Timeline bar with move markers ──
        parts = []
        # Background segments between moves
        boundaries = [0.0] + [m['t_s'] for m in moves] + [dur]
        palette = ['#5AC8FA', '#34C759', '#FF9500', '#AF52DE',
                   '#FF2D55', '#FFCC00', '#5856D6', '#00C7BE']
        for i in range(len(boundaries) - 1):
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            w = max(0.3, (t1 - t0) / dur * 100)
            color = palette[i % len(palette)]
            parts.append(
                f'<div style="position:absolute; left:{t0/dur*100:.2f}%; '
                f'top:0; bottom:0; width:{w:.2f}%; background:{color}; '
                f'opacity:0.4;"></div>')
        # Move markers (vertical lines + clickable timestamps)
        for m in moves:
            x = m['t_s'] / dur * 100
            parts.append(
                f'<div style="position:absolute; left:{x:.2f}%; '
                f'top:0; bottom:0; width:2px; background:#333; '
                f'z-index:1;"></div>')
            parts.append(
                f'<div class="cube-seek" data-time="{m["t_s"]:.3f}" '
                f'style="position:absolute; left:{x:.2f}%; '
                f'top:-14px; transform:translateX(-50%); '
                f'font-size:9px; font-weight:700; color:var(--blue); '
                f'cursor:pointer; white-space:nowrap; text-decoration:underline; '
                f'z-index:2;">{m["t_s"]:.2f}s</div>')
        cube_timeline_el.set_content(''.join(parts))

        # ── Move list with label selectors ──
        rows = [
            '<table style="width:100%; border-collapse:collapse; '
            'font-size:12px; font-family:monospace;">'
            '<thead><tr style="text-align:left; opacity:0.55; '
            'font-weight:700; letter-spacing:0.5px;">'
            '<th style="padding:4px 8px;">#</th>'
            '<th style="padding:4px 8px;">SEGMENT</th>'
            '<th style="padding:4px 8px; text-align:right;">START</th>'
            '<th style="padding:4px 8px; text-align:right;">END</th>'
            '<th style="padding:4px 8px; text-align:right;">DUR</th>'
            '<th style="padding:4px 8px;">MOVE</th>'
            '<th style="padding:4px 8px;">ACTION</th>'
            '</tr></thead><tbody>'
        ]
        # Segments: [0, move0], [move0, move1], ..., [moveN, dur]
        seg_boundaries = [0.0] + [m['t_s'] for m in moves] + [dur]
        for i in range(len(seg_boundaries) - 1):
            t0 = seg_boundaries[i]
            t1 = seg_boundaries[i + 1]
            seg_dur = t1 - t0
            move_val = moves[i]['label'] if i < len(moves) else ''
            color = palette[i % len(palette)]
            rows.append(
                f'<tr style="border-top:1px solid rgba(0,0,0,0.06);">'
                f'<td style="padding:4px 8px; opacity:0.5;">{i+1}</td>'
                f'<td style="padding:4px 8px;">'
                f'<span style="display:inline-block; width:8px; '
                f'height:8px; border-radius:2px; background:{color}; '
                f'margin-right:8px; vertical-align:middle;"></span>'
                f'Segment {i+1}</td>'
                f'<td style="padding:4px 8px; text-align:right;">'
                f'<span class="cube-seek" data-time="{t0:.3f}" '
                f'style="cursor:pointer; color:var(--blue); '
                f'text-decoration:underline;">{t0:6.2f}s</span></td>'
                f'<td style="padding:4px 8px; text-align:right;">'
                f'<span class="cube-seek" data-time="{t1:.3f}" '
                f'style="cursor:pointer; color:var(--blue); '
                f'text-decoration:underline;">{t1:6.2f}s</span></td>'
                f'<td style="padding:4px 8px; text-align:right; '
                f'font-weight:700;">{seg_dur:5.2f}s</td>'
                f'<td style="padding:4px 8px; opacity:0.6;">{move_val}</td>'
                f'<td style="padding:4px 8px;">'
                f'<span class="cube-move-label-select" '
                f'data-seg="{i}" style="cursor:pointer; color:var(--blue); '
                f'text-decoration:underline;">set label</span>'
                f' · '
                f'<span class="cube-move-delete" '
                f'data-seg="{i}" style="cursor:pointer; color:var(--red); '
                f'text-decoration:underline;">delete</span>'
                f'</td></tr>')
        rows.append('</tbody></table>')
        cube_moves_list_el.set_content(''.join(rows))

        cube_status.set_content(
            f'<span style="font-size:12px; color:var(--green);">'
            f'{len(moves)} move(s) detected · '
            f'{len(seg_boundaries)-1} segment(s)</span>')

    def _clear_cube_moves():
        cube_moves.clear()
        _render_cube_moves()
        cube_status.set_content(
            '<span style="font-size:12px; opacity:0.5;">No cube moves</span>')

    def _generate_subtasks_from_moves():
        """Generate subtask segments from cube move timestamps.

        Creates one segment per interval between consecutive moves
        (and from last move to end). Each segment's label defaults
        to empty — user fills via the label selector or manually.
        Clears existing subtasks first.
        """
        moves = cube_moves
        dur = annot_duration[0]
        if not moves or dur <= 0:
            ui.notify('No cube moves loaded or duration unknown',
                      type='warning', position='bottom', timeout=1500)
            return

        annot_subtasks.clear()
        boundaries = [0.0] + [m['t_s'] for m in moves] + [dur]
        for i in range(len(boundaries) - 1):
            t0 = boundaries[i]
            t1 = boundaries[i + 1]
            # Default label: empty (user will assign)
            annot_subtasks.append([float(t0), float(t1), ''])
        _render_subtask_timeline()
        sub_status.set_content(
            f'<span style="font-size:12px; color:var(--green);">'
            f'Generated {len(annot_subtasks)} segments from '
            f'{len(moves)} cube move(s)</span>')
        ui.notify(
            f'Generated {len(annot_subtasks)} segments. '
            'Use the subtask input above to assign labels.',
            type='positive', position='bottom', timeout=3000)

    def _on_cube_move_label_click(seg_idx: int):
        """Open a dialog to pick a subtask label for a cube move segment."""
        moves = cube_moves
        dur = annot_duration[0]
        boundaries = [0.0] + [m['t_s'] for m in moves] + [dur]
        if seg_idx >= len(boundaries) - 1:
            return
        t0 = boundaries[seg_idx]
        t1 = boundaries[seg_idx + 1]

        d = ui.dialog()
        with d, ui.card().style('min-width:400px; padding:20px; gap:12px;'):
            ui.label(f'Assign label to segment {seg_idx+1} '
                     f'({t0:.2f}s → {t1:.2f}s)') \
                .style('font-size:15px; font-weight:700;')
            label_input = ui.input(
                placeholder='e.g. turn the right layer clockwise'
            ).props('dense outlined clearable').style('width:100%;')
            # Preset buttons
            with ui.row().style('gap:6px; flex-wrap:wrap;'):
                for lbl in CUBE_SUBTASK_LABELS:
                    ui.button(lbl, on_click=lambda l=lbl: (
                        setattr(label_input, 'value', l),
                        label_input.update()
                    )).props('outline color="primary" size="sm') \
                        .style('text-transform:none; font-size:11px; '
                               'border-radius:6px;')
            with ui.row().style('justify-content:flex-end; gap:8px;'):
                ui.button('Cancel', on_click=d.close) \
                    .props('flat color="grey"') \
                    .style('text-transform:none;')
                ui.button('Apply', icon='check') \
                    .props('unelevated color="primary"') \
                    .style('text-transform:none; font-weight:600;') \
                    .on('click', lambda: _apply_cube_move_label(
                        seg_idx, label_input.value, d))
        d.open()

    def _apply_cube_move_label(seg_idx: int, label: str, dialog):
        """Apply a label to a cube-move-generated subtask segment."""
        label = (label or '').strip()
        if not label:
            ui.notify('Label cannot be empty', type='warning',
                      position='bottom', timeout=1500)
            return
        if seg_idx < len(annot_subtasks):
            annot_subtasks[seg_idx][2] = label
            _render_subtask_timeline()
            _push_recent(label)
        dialog.close()

    def _on_cube_move_delete(seg_idx: int):
        """Delete a cube move boundary, merging adjacent segments."""
        moves = cube_moves
        if seg_idx >= len(moves):
            return
        moves.pop(seg_idx)
        _render_cube_moves()
        # Also regenerate subtasks if they exist
        if annot_subtasks:
            _generate_subtasks_from_moves()

    async def _save_subtasks():
        cur = annot_current[0]
        if cur is None:
            ui.notify('Pick an episode first', type='warning',
                      position='bottom', timeout=1500)
            return
        if not annot_subtasks:
            ui.notify('No subtasks to save', type='warning',
                      position='bottom', timeout=1500)
            return
        # Recover annot_duration[0] if it's still 0 (mp4 metadata
        # hadn't loaded when the page mounted). Otherwise the
        # *last* subtask segment ends up with t_end=0, and the
        # converter would drop all frames after that segment's
        # start onto the fallback (episode-level) task. The
        # converter has its own t_end=0 → end-of-episode fallback
        # now, but we still want a correct sidecar for any other
        # downstream consumer.
        dur = float(annot_duration[0])
        if dur <= 0.0:
            try:
                d_js = await ui.run_javascript(
                    'const v=document.querySelector('
                    '"video.v4-annot-master, video");'
                    'v && !isNaN(v.duration) ? v.duration : 0',
                    timeout=1.0)
                dur = float(d_js or 0.0)
            except Exception:
                dur = 0.0
        if dur <= 0.0:
            # meta.json fallback (converter wrote n_frames + fps).
            try:
                mp = cur['episode_dir'] / 'meta.json'
                if mp.is_file():
                    m = json.loads(mp.read_text(encoding='utf-8'))
                    n = float(m.get('n_frames', 0) or 0)
                    fps = float(m.get('fps', 0) or 0)
                    if n > 0 and fps > 0:
                        dur = n / fps
            except Exception:
                pass
        if dur <= 0.0 and annot_subtasks:
            # Absolute last resort: pad past the latest segment
            # start so the final segment isn't degenerate.
            dur = max(s[0] for s in annot_subtasks) + 1.0
        annot_duration[0] = dur
        # If the last segment still has end <= start (because it
        # was appended with annot_duration[0]==0 earlier), repair
        # it to `dur` so the JSON we write is self-consistent.
        if annot_subtasks and annot_subtasks[-1][1] <= annot_subtasks[-1][0]:
            annot_subtasks[-1][1] = dur
        payload = {
            'version': 1,
            'duration_s': dur,
            'segments': [
                {'t_start_ns': int(s[0] * 1e9),
                 't_end_ns':   int(s[1] * 1e9),
                 'text':       s[2]}
                for s in annot_subtasks
            ],
            'annotator': app.storage.user.get('username', 'unknown'),
            'annotated_at': datetime.now().isoformat(timespec='seconds'),
        }
        try:
            out_path = cur['episode_dir'] / 'subtasks.json'
            tmp = out_path.with_suffix(out_path.suffix + '.tmp')
            tmp.write_text(json.dumps(payload, ensure_ascii=False,
                                      indent=2),
                           encoding='utf-8')
            os.replace(tmp, out_path)
            sub_status.set_content(
                f'<span style="font-size:12px; color:var(--green); '
                f'font-weight:600;">Saved {len(annot_subtasks)} '
                f'subtasks → {out_path.name}</span>')
            ui.notify(f'Saved {out_path}', type='positive',
                      position='bottom', timeout=2200)
            # Refresh dropdown badges so the green count updates
            # for the just-saved episode without manual Refresh.
            _refresh_badges_keep_selection()
        except Exception as e:
            ui.notify(f'Save subtasks failed: {e}', type='negative',
                      position='bottom', timeout=3000)

    # Per-session job state for the rebake button. Mirrors the
    # convert_job pattern used by the Record page.
    rebake_job = {'proc': None, 'log_lines': deque(maxlen=400)}

    def _rebake_all_annotated():
        """Find sessions with at least one annotated episode (has a
        subtasks.json or contact_modes.json sidecar) and run the
        converter in --rebake-annotations-only mode on each. Streams
        the converter log into a modal dialog so the operator can
        see per-episode progress without dropping to a terminal.
        """
        root = _annot_root_ref[0]
        if not root.exists():
            ui.notify(f'Scan folder not found: {root}', type='warning',
                      position='bottom', timeout=2200)
            return
        # Group: which session dirs contain at least one sidecar?
        sessions_with_annot: list[Path] = []
        n_annotated_eps = 0
        for sub in sorted(root.glob('*_lerobot')):
            if not sub.is_dir():
                continue
            has_any = False
            # Sub may be a session-of-episodes OR a single-bag dir.
            candidates = [sub, *(p for p in sub.iterdir()
                                 if p.is_dir() and
                                 p.name.startswith('episode_'))]
            for ep in candidates:
                if ((ep / 'subtasks.json').is_file()
                        or (ep / 'contact_modes.json').is_file()):
                    n_annotated_eps += 1
                    has_any = True
            if has_any:
                sessions_with_annot.append(sub)
        if not sessions_with_annot:
            ui.notify(
                'No annotated episodes found. Save '
                'subtasks.json (or contact_modes.json) first.',
                type='warning', position='bottom', timeout=2800)
            return

        # ── Modal dialog with tailing log ──
        dlg = ui.dialog().props('persistent').style('z-index:10000;')
        with dlg, ui.card().style(
                'min-width:620px; max-width:820px;'):
            with ui.row().style(
                    'align-items:center; gap:10px; '
                    'margin-bottom:8px;'):
                ui.spinner('dots', size='lg', color='primary')
                ui.html(
                    f'<div style="font-size:16px; '
                    f'font-weight:700;">Re-baking '
                    f'{len(sessions_with_annot)} session(s), '
                    f'{n_annotated_eps} annotated episode(s)…'
                    f'</div>'
                    f'<div style="font-size:12px; opacity:0.5;">'
                    f'Adds <code>task_index</code> column + '
                    f'<code>tasks[]</code> to meta.json. Does '
                    f'NOT re-encode videos.</div>')
            log_el = ui.html('').style(
                'background:#0d1117; color:#c9d1d9; '
                'font-family:monospace; font-size:11px; '
                'padding:12px; border-radius:8px; height:280px; '
                'overflow-y:auto; white-space:pre-wrap; '
                'line-height:1.45;')
            status_el = ui.html(
                '<div style="font-size:12px; opacity:0.6;">'
                'Running…</div>')
            with ui.row().style(
                    'justify-content:flex-end; '
                    'margin-top:10px; gap:8px;'):
                close_btn = ui.button('Close').props(
                    'flat color="grey"').style(
                    'text-transform:none;').on(
                    'click', dlg.close)
                close_btn.set_visibility(False)
        dlg.open()

        # ── Spawn converter ──
        # Repo root is two levels above this file (src/teleop_gui/
        # app.py → wuji-hand-teleop/). Use absolute resolution so
        # the GUI works regardless of where it was launched from.
        repo_root = str(Path(__file__).resolve().parents[2])
        rebake_cmds = ' && '.join(
            f'stdbuf -oL -eL python3 scripts/rosbag2_to_lerobot.py '
            f'"{s}" --rebake-annotations-only'
            for s in sessions_with_annot
        )
        full_cmd = (
            'source /opt/ros/humble/setup.bash 2>/dev/null; '
            'source install/setup.bash 2>/dev/null; '
            + rebake_cmds
        )
        env = os.environ.copy()
        try:
            proc = subprocess.Popen(
                ['bash', '-c', full_cmd],
                env=env, cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except Exception as e:
            status_el.set_content(
                f'<div style="font-size:12px; color:var(--red); '
                f'font-weight:700;">Launch failed: {e}</div>')
            close_btn.set_visibility(True)
            return
        rebake_job['proc'] = proc
        rebake_job['log_lines'].clear()

        def _drain():
            try:
                for line in iter(proc.stdout.readline, ''):
                    rebake_job['log_lines'].append(
                        line.rstrip('\n'))
            except Exception:
                pass
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
        threading.Thread(target=_drain, daemon=True).start()

        rebake_timer_ref = {'t': None}

        def _tick():
            tail = list(rebake_job['log_lines'])[-80:]
            log_el.set_content(
                '<br>'.join(
                    (ln.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;'))
                    for ln in tail) or '—')
            p = rebake_job['proc']
            if p is None:
                t = rebake_timer_ref['t']
                if t is not None:
                    t.deactivate()
                return
            rc = p.poll()
            if rc is None:
                return
            if rc == 0:
                status_el.set_content(
                    '<div style="font-size:12px; '
                    'color:var(--green); font-weight:700;">'
                    '✓ Re-bake complete · '
                    f'{n_annotated_eps} episode(s) baked</div>')
                ui.notify(
                    f'Re-bake done · {len(sessions_with_annot)} '
                    f'session(s)',
                    type='positive', position='bottom',
                    timeout=2800)
                rebake_status.set_content(
                    '<span style="font-size:12px; '
                    'color:var(--green); font-weight:600;">'
                    f'✓ Last bake: {n_annotated_eps} episode(s) '
                    'across '
                    f'{len(sessions_with_annot)} session(s)'
                    '</span>')
            else:
                status_el.set_content(
                    f'<div style="font-size:12px; '
                    f'color:var(--red); font-weight:700;">'
                    f'✗ Re-bake exited with code {rc}</div>')
                ui.notify(f'Re-bake failed (rc={rc})',
                          type='negative',
                          position='bottom', timeout=4000)
            close_btn.set_visibility(True)
            rebake_job['proc'] = None
            t = rebake_timer_ref['t']
            if t is not None:
                t.deactivate()

        rebake_timer_ref['t'] = ui.timer(0.4, _tick)

        def _on_dialog_hide():
            t = rebake_timer_ref['t']
            if t is not None:
                t.deactivate()
        dlg.on('hide', _on_dialog_hide)

    def _on_subtask_recent_picked(_e):
        # Read straight off the select; event arg shape varies
        # across NiceGUI versions.
        v = sub_recent_select.value
        if not v:
            return
        sub_input.value = v
        sub_input.update()
        # Reset select so picking the same option again still fires.
        sub_recent_select.value = None
        try:
            sub_recent_select.update()
        except Exception:
            pass

    sub_recent_select.on('update:model-value',
                          _on_subtask_recent_picked)
    # Enter on the input fires "Add at cursor" — only active when
    # input has focus, so it doesn't conflict with contact-mode 1-6.
    sub_input.on('keydown.enter',
                 lambda: _split_subtask_at_current_time())

    def _on_episode_selected(_e):
        # NiceGUI's update:model-value passes a GenericEventArguments
        # without a .value attribute on some versions; read the
        # current value off the select element directly.
        val = annot_select.value
        if not val:
            return
        picked = next((ep for ep in annot_episodes
                       if ep['label'] == val), None)
        if picked is None:
            return
        # If we're already on this episode_dir (e.g. badges
        # refreshed after a save changed the label string but the
        # underlying episode is the same), skip the re-load —
        # otherwise the video reloads from t=0 every save.
        cur = annot_current[0]
        if (cur is not None
                and cur.get('episode_dir') == picked['episode_dir']):
            cur['label'] = picked['label']  # keep label fresh
            return
        annot_current[0] = picked
        _update_ep_nav()   # keep the "第 X / N 条" indicator in sync
        if callable(on_episode_change):
            try:
                on_episode_change(picked)
            except Exception:
                pass
        annot_segments.clear()
        annot_subtasks.clear()
        existing, dur = _load_existing_annotations(picked['episode_dir'])
        annot_segments.extend(existing)
        existing_subs = _load_existing_subtasks(picked['episode_dir'])
        annot_subtasks.extend(existing_subs)
        # Load cube move timestamps from cube_samples.json (if present).
        cube_moves.clear()
        cube_moves.extend(_load_cube_moves(picked['episode_dir']))
        # Reload detection (bbox) annotators for the new episode (§14).
        for _r in _bbox_reloaders:
            try:
                _r()
            except Exception:
                pass

        # Robust duration fallback: read n_frames + fps from
        # meta.json (the converter always writes these). Used
        # when both the contact-sidecar duration AND the
        # browser-side video.duration are zero / NaN. Without
        # this the colored bar never renders for episodes that
        # only have subtasks.json (no contact_modes.json).
        meta_dur = 0.0
        try:
            mp = picked['episode_dir'] / 'meta.json'
            if mp.is_file():
                m = json.loads(mp.read_text(encoding='utf-8'))
                n = float(m.get('n_frames', 0) or 0)
                fps = float(m.get('fps', 0) or 0)
                if n > 0 and fps > 0:
                    meta_dur = n / fps
        except Exception:
            pass
        # Seed Recent dropdown with this episode's unique subtask texts
        # (most recent at the front).
        seen = set()
        for seg in reversed(existing_subs):
            txt = seg[2]
            if txt and txt not in seen:
                seen.add(txt)
                _push_recent(txt)
        # Set sources on all three views. cam_urls may be partial
        # (some episodes only have 1-2 cams) — fall back to the
        # master URL or empty so set_source never crashes.
        cam_urls = picked.get('cam_urls') or {}
        master_url = (cam_urls.get('global')
                      or picked['video_url'])
        annot_video.set_source(master_url)
        annot_video_lwrist.set_source(
            cam_urls.get('left_wrist', ''))
        annot_video_rwrist.set_source(
            cam_urls.get('right_wrist', ''))
        # Ask the browser for duration, fall back to meta.json /
        # any saved value. Priority: live JS duration > meta.json
        # (n_frames/fps) > contact-sidecar saved dur. Even if the
        # JS query times out or returns 0 we still get a real
        # duration so the timeline renders.
        async def _after_load():
            # HTML5 <video> does NOT reload when its bound `src`
            # attribute changes — without an explicit .load() the
            # element can keep showing the PREVIOUS episode's buffered
            # frames (intermittent "video ≠ this episode"). Wait for Vue
            # to patch the new src into the DOM, then force .load() on
            # all three views. Selector is doubled (class may sit on the
            # <video> OR a wrapper) so the wrist ("slave") players are
            # included too.
            await asyncio.sleep(0.15)
            try:
                await ui.run_javascript(
                    "document.querySelectorAll("
                    "'video.v4-annot-video, .v4-annot-video video')"
                    ".forEach(v => { try { v.load(); } catch (e) {} });")
            except Exception:
                pass
            await asyncio.sleep(0.3)
            d_js = 0.0
            try:
                d_js = await ui.run_javascript(
                    'const v = document.querySelector("video.v4-annot-master, video");'
                    'v && !isNaN(v.duration) ? v.duration : 0',
                    timeout=1.0)
                d_js = float(d_js or 0.0)
            except Exception:
                d_js = 0.0
            annot_duration[0] = max(d_js, meta_dur, dur)
            # Heal any in-memory subtask segments whose t_end <=
            # t_start (artifact of a pre-fix save where the
            # video.duration was still NaN — see git log
            # 2026-05-18 fix). After this, the bar + table both
            # render correctly, AND if the operator saves again
            # the json gets the proper t_end_ns.
            if annot_duration[0] > 0:
                for s in annot_subtasks:
                    if s[1] <= s[0]:
                        s[1] = float(annot_duration[0])
            _render_timeline()
            _render_subtask_timeline()
            _render_cube_moves()
            annot_time_label.set_content(
                f'<div style="font-size:11px; opacity:0.4; font-family:monospace; margin-top:4px;">'
                f'{0.0:.2f}s / {annot_duration[0]:.2f}s</div>')
            annot_status.set_content(
                f'<span style="font-size:12px;">Loaded · '
                f'{len(annot_segments)} contact · '
                f'{len(annot_subtasks)} subtasks · '
                f'{annot_duration[0]:.1f}s</span>')
            if annot_subtasks:
                sub_status.set_content(
                    f'<span style="font-size:12px;">Loaded '
                    f'{len(annot_subtasks)} subtasks</span>')
            else:
                sub_status.set_content(
                    '<span style="font-size:12px; opacity:0.5;">'
                    'No subtasks yet</span>')
        asyncio.create_task(_after_load())

    annot_select.on('update:model-value', _on_episode_selected)

    def _update_ep_nav():
        """Refresh the "第 X / N 条" indicator next to the prev/next
        buttons for the currently-loaded episode."""
        try:
            cur = annot_current[0]
            total = len(annot_episodes)
            idx = -1
            if cur is not None:
                cd = cur.get('episode_dir')
                idx = next((i for i, ep in enumerate(annot_episodes)
                            if ep.get('episode_dir') == cd), -1)
            ep_nav_pos.text = (f'第 {idx + 1} / {total} 条'
                               if idx >= 0 else f'共 {total} 条')
        except Exception:
            pass

    def _go_episode(delta: int):
        """Jump to the prev (-1) / next (+1) episode in the dropdown,
        loading it via the normal selection path."""
        if not annot_episodes:
            return
        cur = annot_current[0]
        idx = -1
        if cur is not None:
            cd = cur.get('episode_dir')
            idx = next((i for i, ep in enumerate(annot_episodes)
                        if ep.get('episode_dir') == cd), -1)
        new_idx = (idx if idx >= 0 else 0) + delta
        if new_idx < 0:
            ui.notify('已经是第一条 episode', type='info',
                      position='bottom', timeout=1500)
            return
        if new_idx >= len(annot_episodes):
            ui.notify('已经是最后一条 episode', type='info',
                      position='bottom', timeout=1500)
            return
        # Reassign options (same strings) to force the q-select to
        # re-render its displayed value — setting .value alone doesn't
        # reliably sync the dropdown display in this NiceGUI version.
        annot_select.options = [ep['label'] for ep in annot_episodes]
        annot_select.value = annot_episodes[new_idx]['label']
        annot_select.update()
        _on_episode_selected(None)

    # Poll the current video time for the label (1 Hz — cheap, only
    # on Annotate page).
    async def _tick_annot_time():
        if not _act() or annot_current[0] is None:
            return
        t = await _get_video_time_async()
        d = annot_duration[0]
        annot_time_label.set_content(
            f'<div style="font-size:11px; opacity:0.4; font-family:monospace; margin-top:4px;">'
            f'{t:.2f}s / {d:.2f}s</div>')
    ui.timer(0.5, _tick_annot_time)

    # Keyboard shortcuts — only active when this page is visible.
    def _on_key(e):
        if not _act():
            return
        # Only handle on keydown and ignore modifier-only keys
        if not getattr(e, 'action', None) or not e.action.keydown:
            return
        key = getattr(e.key, 'name', None) or str(e.key)
        # 数字键 1..9,0 → subtask 预设速填:改由文档级 JS 捕获监听处理(见
        # add_body_html),这样焦点在 <video>/视频轴 上也生效,并挡掉视频原生跳转。
        if key in ('Backspace', 'Delete'):
            _undo_last_segment()
            return
        if key in ('s', 'S'):
            _save_annotations()
            return
        if key == ' ':
            ui.run_javascript(
                'const v = document.querySelector("video.v4-annot-master, video");'
                'if (v) { v.paused ? v.play() : v.pause(); }')
            return
        if key in ('ArrowLeft', 'ArrowRight'):
            step = 0.1 if (e.modifiers and e.modifiers.shift) else 0.5
            sign = -1 if key == 'ArrowLeft' else 1
            ui.run_javascript(
                f'const v = document.querySelector("video.v4-annot-master, video");'
                f'if (v) v.currentTime = Math.max(0, v.currentTime + ({sign * step}));')
            return

    ui.keyboard(on_key=_on_key)

    # 数字键预设:JS 捕获监听 → emitEvent('subtask_preset', '<digit>') → 这里应用。
    # 好处是焦点在 <video>/视频轴 上也能用(ui.keyboard 会被视频原生行为抢走)。
    _PRESET_IDX = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
                   '6': 5, '7': 6, '8': 7, '9': 8, '0': 9}

    def _on_preset_key(e):
        if not _act():
            return
        k = e.args
        if isinstance(k, dict):
            k = k.get('digit') or k.get('key') or ''
        idx = _PRESET_IDX.get(str(k))
        if idx is not None:
            _apply_preset_subtask(idx)
    ui.on('subtask_preset', _on_preset_key)

    # ── Cube move label/delete event handlers (JS → Python) ──
    def _on_cube_label_click(e):
        if not _act():
            return
        seg = (e.args or {}).get('seg')
        if seg is not None:
            _on_cube_move_label_click(int(seg))

    def _on_cube_delete_click(e):
        if not _act():
            return
        seg = (e.args or {}).get('seg')
        if seg is not None:
            _on_cube_move_delete(int(seg))

    ui.on('cube_move_label', _on_cube_label_click)
    ui.on('cube_move_delete', _on_cube_delete_click)

    def _rescan_and_repop():
        annot_episodes.clear()
        annot_episodes.extend(_scan_lerobot_episodes())
        annot_select.options = [ep['label'] for ep in annot_episodes]
        annot_select.update()
        if not annot_episodes:
            # Show empty-state gate, hide working area.
            annot_gate_panel.style('display:block;')
            annot_work_panel.style('display:none;')
            annot_status.set_content(
                '<span style="font-size:12px; color:var(--amber);">'
                'No converted episodes yet.</span>')
        else:
            annot_gate_panel.style('display:none;')
            annot_work_panel.style('display:block;')
            annot_status.set_content(
                f'<span style="font-size:12px;">{len(annot_episodes)} episode(s) '
                f'available</span>')

    def _refresh_badges_keep_selection():
        """Rescan to update 🔴/🟢 counts in the dropdown labels,
        but keep the currently-loaded episode selected so the
        video doesn't reload to t=0. Used after Save subtasks /
        Save contact_modes so the badges update inline.

        The `_on_episode_selected` guard catches the value-change
        event we trigger here and short-circuits because the
        episode_dir is unchanged."""
        cur = annot_current[0]
        cur_dir = cur['episode_dir'] if cur is not None else None
        _rescan_and_repop()
        if cur_dir is None:
            return
        match = next(
            (ep for ep in annot_episodes
             if ep.get('episode_dir') == cur_dir),
            None)
        if match is None:
            return
        # Keep the cached current dict's label in sync too so
        # later comparisons (e.g. dropdown re-renders) still
        # match.
        annot_current[0] = match
        annot_select.value = match['label']
        annot_select.update()

    # Initial scan
    _rescan_and_repop()
