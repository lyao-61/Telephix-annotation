#!/usr/bin/env python3
"""Media-root registration + episode preview-mp4 generation.

Extracted byte-for-byte from app.py (v3 refactor). The module-level
`_MEDIA_ROOTS` registry and the `_PREVIEW_*` constants travel with the
functions that own them, so the registry state stays a singleton scoped
to these helpers. `_register_media_root` mounts dirs on the shared NiceGUI
`app` for static serving; `_generate_preview_mp4` keeps its heavy deps
(cv2 / numpy / rosbag2_py / rclpy) as function-local imports exactly as
in the original.
"""
import os
from pathlib import Path

from nicegui import app

# Media-root registry. Browsers can only fetch files under paths that
# FastAPI has mounted as static, but the Record-page scan root is
# user-configurable and may sit outside ~/teleop_bags (e.g. a sibling
# ZWHand bag dir). Each unique root gets its own /teleop-media[-N]
# mount; preview/annotate URL builders look up the right prefix here.
_MEDIA_ROOTS: list[tuple[Path, str]] = []   # (resolved_root, url_prefix)


def _register_media_root(root: Path, prefix: str | None = None) -> str:
    """Mount `root` for static serving and return its URL prefix.

    Idempotent: if `root` (or a parent of it) is already mounted, the
    existing prefix is returned without remounting. `add_media_files`
    edits FastAPI routes; calling it twice for the same path leaves
    duplicate routes that resolve to the same dir but waste lookups.
    """
    try:
        root = Path(root).resolve()
    except Exception:
        return ''
    for p, pref in _MEDIA_ROOTS:
        try:
            root.relative_to(p)
            return pref
        except ValueError:
            continue
    if prefix is None:
        prefix = f'/teleop-media-{len(_MEDIA_ROOTS)}'
    try:
        os.makedirs(root, exist_ok=True)
        app.add_media_files(prefix, str(root))
    except Exception:
        return ''
    _MEDIA_ROOTS.append((root, prefix))
    return prefix


def _media_url(path: Path) -> str | None:
    """Return a browser-fetchable URL for `path`, or None if no mount covers it."""
    try:
        abs_p = Path(path).resolve()
    except Exception:
        return None
    for root, prefix in _MEDIA_ROOTS:
        try:
            rel = abs_p.relative_to(root)
            return f'{prefix}/{rel.as_posix()}'
        except ValueError:
            continue
    return None


# Reads 3 camera streams from a single episode bag, builds a side-by-side
# grid (left_wrist | global | right_wrist), writes a low-res 10fps mp4
# cached in the episode folder as `preview.mp4`. Idempotent — returns
# immediately if the file already exists. Designed to run in a worker
# thread so the UI stays responsive. ~3-5 s per 20 s episode on a
# modern CPU with OpenCV's mp4v encoder.
_PREVIEW_TOPICS = (
    ('/camera/left_wrist/left_wrist_camera/color/image_raw', 'left'),
    ('/camera/global/global_camera/color/image_raw',        'global'),
    ('/camera/right_wrist/right_wrist_camera/color/image_raw', 'right'),
)
_PREVIEW_CELL_W = 320
_PREVIEW_CELL_H = 240
_PREVIEW_FPS = 10   # every 3rd frame from 30 Hz source


def _generate_preview_mp4(episode_dir, force: bool = False):
    """Build `<episode_dir>/preview.mp4` if missing. Returns the Path or None.

    Uses the system ffmpeg (libx264) rather than OpenCV's VideoWriter:
    OpenCV's bundled ffmpeg tries h264_v4l2m2m (hardware encoder) and
    falls back to mp4v (MPEG-4 Part 2) which Chrome cannot play inline
    — user sees the video element with controls but a black frame. H.264
    yuv420p with faststart is the browser-universal combo.
    """
    import subprocess
    from pathlib import Path as _P
    episode_dir = _P(episode_dir)
    preview = episode_dir / 'preview.mp4'
    if preview.exists() and not force:
        return preview
    dbs = list(episode_dir.glob('*.db3'))
    if not dbs:
        return None
    try:
        import cv2
        import numpy as np
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except Exception as e:
        raise RuntimeError(f'preview deps missing: {e}')

    # Read frames per-topic (stored in BGR so ffmpeg -pix_fmt bgr24 works).
    topic_to_side = dict(_PREVIEW_TOPICS)
    frames = {'left': [], 'global': [], 'right': []}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(episode_dir), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t, _ in _PREVIEW_TOPICS]))
    while reader.has_next():
        topic, data, ts = reader.read_next()
        side = topic_to_side.get(topic)
        if side is None:
            continue
        msg = deserialize_message(data, Image)
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            arr = arr[:, :, ::-1]   # RGB → BGR
        arr = cv2.resize(arr, (_PREVIEW_CELL_W, _PREVIEW_CELL_H))
        frames[side].append((ts, arr))

    if not frames['left'] or not frames['global'] or not frames['right']:
        return None

    for k in frames:
        frames[k].sort(key=lambda x: x[0])
    n = min(len(frames['left']), len(frames['global']), len(frames['right']))
    step = max(1, 30 // _PREVIEW_FPS)

    grid_w = _PREVIEW_CELL_W * 3
    grid_h = _PREVIEW_CELL_H

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{grid_w}x{grid_h}',
        '-r', str(_PREVIEW_FPS),
        '-i', '-',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-crf', '28',
        '-movflags', '+faststart',
        str(preview),
    ]
    try:
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError('ffmpeg not installed (apt install ffmpeg)')
    try:
        for i in range(0, n, step):
            cells = [frames[s][i][1] for s in ('left', 'global', 'right')]
            grid = np.hstack(cells)
            if not grid.flags['C_CONTIGUOUS']:
                grid = np.ascontiguousarray(grid)
            proc.stdin.write(grid.tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait(timeout=60)
    if rc != 0:
        err = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''
        raise RuntimeError(f'ffmpeg exit {rc}: {err[:400]}')
    return preview if preview.exists() else None
