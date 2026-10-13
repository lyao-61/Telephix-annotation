"""Native folder picker (zenity) — copied from teleop's record panel.

Returns the picked path, '' on cancel, or the sentinel 'NO_ZENITY' when zenity
is not installed. The blocking subprocess runs in a worker thread so the event
loop stays responsive. The dialog opens on the machine running this app.
"""
from __future__ import annotations

import asyncio
import os
import subprocess


async def _pick_directory_dialog(initial: str = '',
                                 title: str = 'Select Folder') -> str | None:
    def _run():
        try:
            cmd = ['zenity', '--file-selection', '--directory', '--title', title]
            if initial:
                start = os.path.expanduser(initial).rstrip('/') + '/'
                cmd += ['--filename', start]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                return result.stdout.strip()
            return ''
        except FileNotFoundError:
            return 'NO_ZENITY'
        except Exception:
            return ''
    return await asyncio.to_thread(_run)
