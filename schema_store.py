"""Annotation-schema store (self-contained, JSON).

A "schema" (kept as a project dict for compatibility with the annotate stage)
defines WHICH fields to annotate and their types:

    { "id", "name", "created_at", "updated_at", "next_ann_seq",
      "annotations": [ {"id","name","type","required","description",
                        "options","dim"} ] }

type ∈ {string(subtask 长时程语言), int, float, bool, detection(检测框 bbox)}.

Storage: <data_dir>/projects.json  (data_dir = $TELEPHIX_ANNOTATION_DIR or
~/telephix_annotation). Atomic writes. No dependency on teleop.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

ANN_TYPES = {
    'string': '字符串 / subtask 长时程语言',
    'int': '整数 (int)',
    'float': '浮点数 (float)',
    'bool': '布尔 (bool)',
    'detection': '检测框 (detection · bbox)',
}


def data_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get('TELEPHIX_ANNOTATION_DIR',
                                               '~/telephix_annotation')))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file() -> Path:
    return data_dir() / 'projects.json'


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def valid_field_name(name: str) -> bool:
    return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', (name or '').strip()))


def _atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def load_projects() -> list[dict]:
    f = _file()
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return sorted(data, key=lambda p: p.get('updated_at', ''), reverse=True)


def get_project(pid: str) -> dict | None:
    return next((p for p in load_projects() if p.get('id') == pid), None)


def name_taken(name: str, exclude_id: str | None = None) -> bool:
    n = ' '.join((name or '').split()).casefold()
    return any(' '.join((p.get('name', '')).split()).casefold() == n
               and p.get('id') != exclude_id for p in load_projects())


def new_project_id() -> str:
    base = 'sch_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    existing = {p.get('id') for p in load_projects()}
    if base not in existing:
        return base
    i = 2
    while f'{base}_{i}' in existing:
        i += 1
    return f'{base}_{i}'


def alloc_ann_id(project: dict) -> str:
    n = project.get('next_ann_seq')
    if n is None:
        mx = 0
        for a in project.get('annotations', []) or []:
            aid = a.get('id') or ''
            if aid.startswith('ann_'):
                try:
                    mx = max(mx, int(aid[4:]))
                except ValueError:
                    pass
        n = mx + 1
    project['next_ann_seq'] = n + 1
    return f'ann_{n}'


def save_project(project: dict) -> dict:
    if not project.get('id'):
        project['id'] = new_project_id()
    name = (project.get('name') or '').strip()
    if not name:
        raise ValueError('方案名称不能为空')
    if name_taken(name, exclude_id=project['id']):
        raise ValueError(f'方案名称已存在：{name}')
    project['name'] = name
    seen: set[str] = set()
    for a in project.get('annotations', []) or []:
        nm = (a.get('name') or '').strip()
        if not nm:
            continue
        if not valid_field_name(nm):
            raise ValueError(f'标注字段名非法：{nm}（仅字母/数字/下划线，字母或下划线开头）')
        if nm in seen:
            raise ValueError(f'标注字段名重复：{nm}')
        seen.add(nm)
        if not a.get('id'):
            a['id'] = alloc_ann_id(project)
        a.setdefault('required', False)
        a.setdefault('description', '')
        a.setdefault('options', [])
        a.setdefault('dim', None)
    project.setdefault('created_at', _now())
    project['updated_at'] = _now()
    projects = load_projects()
    for i, p in enumerate(projects):
        if p.get('id') == project['id']:
            projects[i] = project
            break
    else:
        projects.append(project)
    _atomic_write(_file(), projects)
    return project


def delete_project(pid: str) -> None:
    _atomic_write(_file(), [p for p in load_projects() if p.get('id') != pid])
