"""Trivial login guard. Annotator (``username``) and collector (``collector``)
names are stored per-browser; the annotator goes into every sidecar's
``annotator`` field, and both are added to the folder name on transfer."""
from __future__ import annotations

from nicegui import app, ui


def require_login() -> bool:
    u = app.storage.user
    if not (u.get('username') and u.get('collector')):
        ui.navigate.to('/')
        return False
    return True
