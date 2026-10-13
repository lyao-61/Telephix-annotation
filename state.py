"""Minimal shared state for the standalone annotation tool.

The annotate submodules (annotate_stage / project_annotate / bbox / value)
only read ``S.active_project`` — the currently-selected annotation schema
(a dict with an ``annotations`` field-definition list).
"""
from __future__ import annotations


class State:
    def __init__(self) -> None:
        self.active_project: dict | None = None


S = State()
