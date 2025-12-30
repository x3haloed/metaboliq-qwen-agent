# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple, Union

from qwen_agent.tools.base import BaseTool, register_tool

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

Selector = Union[str, List[Union[str, int]], Tuple[int, Union[int, str]]]
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_CHARS = 4000


def _detect_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.py':
        return 'tree'
    if ext in ('.json', '.yaml', '.yml'):
        return 'map'
    if ext in ('.csv', '.tsv'):
        return 'table'
    return 'blob'


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _load_map(path: str) -> Any:
    ext = os.path.splitext(path)[1].lower()
    content = _read_text(path)
    if ext == '.json':
        return json.loads(content)
    if ext in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('yaml is required for .yaml/.yml support: pip install pyyaml')
        return yaml.safe_load(content)
    raise ValueError('Unsupported map format')


def _write_map(path: str, data: Any) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        _write_text(path, json.dumps(data, indent=2))
        return
    if ext in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError('yaml is required for .yaml/.yml support: pip install pyyaml')
        _write_text(path, yaml.safe_dump(data, sort_keys=False))
        return
    raise ValueError('Unsupported map format')


def _read_table(path: str):
    ext = os.path.splitext(path)[1].lower()
    delimiter = '\t' if ext == '.tsv' else ','
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _write_table(path: str, header: List[str], rows: List[List[str]]) -> None:
    ext = os.path.splitext(path)[1].lower()
    delimiter = '\t' if ext == '.tsv' else ','
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(rows)


def _outline_tree(path: str) -> Dict[str, Any]:
    source = _read_text(path)
    tree = ast.parse(source)
    functions = []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {
        'summary': 'tree',
        'functions': functions,
        'classes': classes,
    }


def _select_tree(path: str, selector: str) -> str:
    source = _read_text(path)
    tree = ast.parse(source)
    kind, name = selector.split(':', 1)
    for node in tree.body:
        if kind == 'function' and isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            return segment or ast.unparse(node)
        if kind == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            return segment or ast.unparse(node)
    raise KeyError(f'{selector} not found')


def _replace_tree(path: str, selector: str, value: str) -> None:
    source = _read_text(path)
    tree = ast.parse(source)
    kind, name = selector.split(':', 1)
    for node in tree.body:
        if kind == 'function' and isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            lines = source.splitlines(keepends=True)
            replacement = value if value.endswith('\n') else value + '\n'
            _write_text(path, ''.join(lines[:start]) + replacement + ''.join(lines[end:]))
            return
        if kind == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            lines = source.splitlines(keepends=True)
            replacement = value if value.endswith('\n') else value + '\n'
            _write_text(path, ''.join(lines[:start]) + replacement + ''.join(lines[end:]))
            return
    raise KeyError(f'{selector} not found')


def _safe_json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=True))
    except Exception:
        return len(str(value))


def _paginate_list(items: List[Any], page: int, page_size: int) -> Dict[str, Any]:
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = DEFAULT_PAGE_SIZE
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    sliced = items[start:end]
    next_page = page + 1 if end < total else None
    return {
        'items': sliced,
        'page': page,
        'page_size': page_size,
        'total': total,
        'truncated': end < total,
        'next_page': next_page,
    }


def _paginate_text(text: str, page: int, page_size: int) -> Dict[str, Any]:
    lines = text.splitlines()
    page_info = _paginate_list(lines, page, page_size)
    page_text = '\n'.join(page_info['items'])
    return {
        'text': page_text,
        'page': page_info['page'],
        'page_size': page_info['page_size'],
        'total': page_info['total'],
        'truncated': page_info['truncated'],
        'next_page': page_info['next_page'],
    }


@register_tool('describe_file')
class DescribeFile(BaseTool):
    description = 'Describe a file structure (like a lightweight, shape-aware cat). Accepts absolute local paths.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file.'
            },
            'page': {
                'type': 'integer',
                'description': 'Optional page number (1-based) for long outlines.'
            },
            'page_size': {
                'type': 'integer',
                'description': 'Optional page size for long outlines.'
            },
        },
        'required': ['path'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        params = self._verify_json_format_args(params)
        path = params['path']
        page = params.get('page', 1)
        page_size = params.get('page_size', DEFAULT_PAGE_SIZE)
        kind = _detect_kind(path)
        if kind == 'tree':
            return {'kind': kind, 'outline': _outline_tree(path)}
        if kind == 'map':
            data = _load_map(path)
            if isinstance(data, dict):
                keys = list(data.keys())
                page_info = _paginate_list(keys, page, page_size)
                outline = {
                    'summary': 'map',
                    'keys': page_info['items'],
                }
                if page_info['truncated']:
                    outline['note'] = (
                        f'Keys truncated. Call describe_file with page={page_info["next_page"]} '
                        f'to continue.'
                    )
                outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
                return {'kind': kind, 'outline': outline}
            if isinstance(data, list):
                return {'kind': kind, 'outline': {'summary': 'map-list', 'length': len(data)}}
            return {'kind': kind, 'outline': {'summary': 'map-scalar', 'type': type(data).__name__}}
        if kind == 'table':
            if pd is not None:
                df = pd.read_csv(path)
                page_info = _paginate_list(list(range(len(df))), page, page_size)
                start = (page_info['page'] - 1) * page_info['page_size']
                end = start + page_info['page_size']
                return {
                    'kind': kind,
                    'outline': {
                        'summary': 'table',
                        'row_count': len(df),
                        'columns': list(df.columns),
                        'head': df.iloc[start:end].to_dict(orient='records'),
                        'page': page_info['page'],
                        'page_size': page_info['page_size'],
                        'total': page_info['total'],
                        'truncated': page_info['truncated'],
                        'next_page': page_info['next_page'],
                        'note': (
                            f'Rows truncated. Call describe_file with page={page_info["next_page"]} '
                            f'to continue.' if page_info['truncated'] else ''
                        ),
                    },
                }
            header, rows = _read_table(path)
            page_info = _paginate_list(rows, page, page_size)
            return {
                'kind': kind,
                'outline': {
                    'summary': 'table',
                    'row_count': len(rows),
                    'columns': header,
                    'head': page_info['items'],
                    'page': page_info['page'],
                    'page_size': page_info['page_size'],
                    'total': page_info['total'],
                    'truncated': page_info['truncated'],
                    'next_page': page_info['next_page'],
                    'note': (
                        f'Rows truncated. Call describe_file with page={page_info["next_page"]} '
                        f'to continue.' if page_info['truncated'] else ''
                    ),
                },
            }
        with open(path, 'rb') as f:
            blob = f.read()
        return {
            'kind': kind,
            'outline': {
                'summary': 'blob',
                'size': len(blob),
                'sha256': hashlib.sha256(blob).hexdigest(),
            },
        }


@register_tool('extract_section')
class ExtractSection(BaseTool):
    description = 'Extract a specific section of a file using a shape-aware selector. Accepts absolute local paths.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file.'
            },
            'selector': {
                'description': 'Tree: "function:<name>" or "class:<name>". Map: ["a", 0, "b"]. Table: [row, col].'
            },
            'page': {
                'type': 'integer',
                'description': 'Optional page number (1-based) for large values.'
            },
            'page_size': {
                'type': 'integer',
                'description': 'Optional page size for large values.'
            },
        },
        'required': ['path', 'selector'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Any:
        params = self._verify_json_format_args(params)
        path = params['path']
        selector = params['selector']
        page = params.get('page', 1)
        page_size = params.get('page_size', DEFAULT_PAGE_SIZE)
        kind = _detect_kind(path)
        if kind == 'tree':
            if not isinstance(selector, str):
                raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
            text = _select_tree(path, selector)
            page_info = _paginate_text(text, page, page_size)
            value = page_info['text']
            resp = {'kind': kind, 'value': value}
            resp.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
            if page_info['truncated']:
                resp['note'] = (
                    f'Text truncated. Call extract_section with page={page_info["next_page"]} '
                    f'to continue.'
                )
            return resp
        if kind == 'map':
            if not isinstance(selector, list):
                raise ValueError('Map selector must be a list path')
            data = _load_map(path)
            obj = data
            for key in selector:
                obj = obj[key]
            if isinstance(obj, dict):
                keys = list(obj.keys())
                if len(keys) <= page_size and _safe_json_size(obj) <= DEFAULT_MAX_CHARS:
                    return {'kind': kind, 'value': obj}
                page_info = _paginate_list(keys, page, page_size)
                resp = {'kind': kind, 'value': page_info['items']}
                resp.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
                if page_info['truncated']:
                    resp['note'] = (
                        f'Keys truncated. Call extract_section with page={page_info["next_page"]} '
                        f'to continue, or select a deeper path.'
                    )
                return resp
            if isinstance(obj, list):
                if len(obj) <= page_size and _safe_json_size(obj) <= DEFAULT_MAX_CHARS:
                    return {'kind': kind, 'value': obj}
                page_info = _paginate_list(obj, page, page_size)
                resp = {'kind': kind, 'value': page_info['items']}
                resp.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
                if page_info['truncated']:
                    resp['note'] = (
                        f'List truncated. Call extract_section with page={page_info["next_page"]} '
                        f'to continue.'
                    )
                return resp
            if isinstance(obj, str):
                page_info = _paginate_text(obj, page, page_size)
                resp = {'kind': kind, 'value': page_info['text']}
                resp.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
                if page_info['truncated']:
                    resp['note'] = (
                        f'Text truncated. Call extract_section with page={page_info["next_page"]} '
                        f'to continue.'
                    )
                return resp
            return {'kind': kind, 'value': obj}
        if kind == 'table':
            if not isinstance(selector, list) or len(selector) != 2:
                raise ValueError('Table selector must be [row_index, column]')
            row, col = selector
            if pd is not None:
                df = pd.read_csv(path)
                if isinstance(col, int):
                    col = df.columns[col]
                return {'kind': kind, 'value': df.at[row, col]}
            header, rows = _read_table(path)
            if isinstance(col, int):
                col_idx = col
            else:
                col_idx = header.index(col)
            return {'kind': kind, 'value': rows[row][col_idx]}
        raise ValueError('select not supported for blob')


@register_tool('replace_section')
class ReplaceSection(BaseTool):
    description = 'Replace a specific section of a file using a shape-aware selector. Accepts absolute local paths.'
    parameters = {
        'type': 'object',
        'properties': {
            'path': {
                'type': 'string',
                'description': 'Path to the file.'
            },
            'selector': {
                'description': 'Tree: "function:<name>" or "class:<name>". Map: ["a", 0, "b"]. Table: [row, col].'
            },
            'value': {
                'description': 'Replacement value or source code (for tree).'
            },
        },
        'required': ['path', 'selector', 'value'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict[str, Any]:
        params = self._verify_json_format_args(params)
        path = params['path']
        selector = params['selector']
        value = params['value']
        kind = _detect_kind(path)
        if kind == 'tree':
            if not isinstance(selector, str):
                raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
            if not isinstance(value, str):
                raise ValueError('Tree replacement value must be source code')
            _replace_tree(path, selector, value)
            return {'changed': True, 'kind': kind}
        if kind == 'map':
            if not isinstance(selector, list):
                raise ValueError('Map selector must be a list path')
            data = _load_map(path)
            obj = data
            for key in selector[:-1]:
                obj = obj[key]
            obj[selector[-1]] = value
            _write_map(path, data)
            return {'changed': True, 'kind': kind}
        if kind == 'table':
            if not isinstance(selector, list) or len(selector) != 2:
                raise ValueError('Table selector must be [row_index, column]')
            row, col = selector
            if pd is not None:
                df = pd.read_csv(path)
                if isinstance(col, int):
                    col = df.columns[col]
                df.at[row, col] = value
                df.to_csv(path, index=False)
                return {'changed': True, 'kind': kind}
            header, rows = _read_table(path)
            if isinstance(col, int):
                col_idx = col
            else:
                col_idx = header.index(col)
            rows[row][col_idx] = value
            _write_table(path, header, rows)
            return {'changed': True, 'kind': kind}
        raise ValueError('replace not supported for blob')
