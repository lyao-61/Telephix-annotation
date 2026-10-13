"""Visual rsync transfer — Data Studio owned.

Wraps ``rsync -avh --partial --info=progress2`` (equivalent to the user's
``rsync -avhP`` but with an overall-percent line that's easy to parse for a
progress bar). Password is passed to ssh via ``sshpass -e`` (SSHPASS env var),
so it never appears in the process argument list.

Requires ``rsync`` and ``sshpass`` on the local machine.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil

_PCT = re.compile(r'(\d+)%')
_UNSAFE = re.compile(r'[\s/\\:*?"<>|]+')


def _safe(name: str) -> str:
    return _UNSAFE.sub('-', (name or '').strip()).strip('-') or 'unknown'


def tagged_name(folder: str, collector: str, annotator: str) -> str:
    """Folder name with the collector/annotator names added. Kept in front of a
    trailing ``_lerobot`` so the receiving side still scans it as a session:
    ``session_x_lerobot`` → ``session_x_采集-张三_标注-李四_lerobot``."""
    tag = f"采集-{_safe(collector)}_标注-{_safe(annotator)}"
    suffix = "_lerobot" if folder.endswith("_lerobot") else ""
    base = folder[:len(folder) - len(suffix)]
    if base.endswith(tag):      # already tagged — don't add twice
        return folder
    return f"{base}_{tag}{suffix}"


def preflight() -> list[str]:
    """Return names of required tools that are missing (empty = ready)."""
    return [t for t in ("rsync", "sshpass") if not shutil.which(t)]


_SSH = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"


def build(local: str, user_host: str, remote_path: str,
          contents_only: bool) -> list[str]:
    """rsync argv for a single local folder (sans password → SSHPASS env).
    `contents_only` appends a trailing slash so the folder's *contents* land in
    `remote_path`."""
    src = os.path.expanduser((local or "").rstrip("/"))
    if contents_only:
        src += "/"
    dest = f"{user_host}:{remote_path}"
    return ["sshpass", "-e", "rsync", "-avh", "--partial",
            "--info=progress2", "-e", _SSH, src, dest]


def build_relative(sources: list[str], user_host: str,
                   remote_path: str) -> list[str]:
    """rsync argv for many episode dirs, using ``--relative`` + a ``/./`` marker
    in each source so it lands as ``<remote_path>/<session>/<episode>/``."""
    dest = f"{user_host}:{remote_path}"
    return ["sshpass", "-e", "rsync", "-avh", "--partial",
            "--info=progress2", "--relative", "-e", _SSH, *sources, dest]


async def stream(argv: list[str], password: str, on_progress, on_line) -> int:
    """Run a prebuilt rsync `argv`, streaming output. Calls `on_progress(int_pct)`
    and `on_line(str)` as rsync emits. Returns the process exit code.

    rsync updates progress with carriage returns, so we read raw chunks and
    split on both \\r and \\n. stderr is merged into stdout for the log.
    """
    env = dict(os.environ, SSHPASS=password or "")
    proc = await asyncio.create_subprocess_exec(
        *argv, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    buf = ""
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", "replace")
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()               # keep the incomplete tail
        for line in parts:
            line = line.strip()
            if not line:
                continue
            m = _PCT.search(line)
            if m:
                try:
                    on_progress(min(100, int(m.group(1))))
                except Exception:
                    pass
            on_line(line)
    if buf.strip():
        on_line(buf.strip())
    return await proc.wait()
