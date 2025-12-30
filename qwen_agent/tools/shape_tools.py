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

import os
from typing import Any, Dict, List, Tuple, Union

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.shape_handlers import get_handler, supported_extensions
from qwen_agent.tools.shape_handlers.common import DEFAULT_PAGE_SIZE

Selector = Union[str, List[Union[str, int]], Tuple[int, Union[int, str]]]


def _parse_selector(selector: Any) -> Selector:
    if isinstance(selector, (list, tuple)):
        return selector
    if not isinstance(selector, str):
        raise ValueError('Selector must be a list path or a string path')
    selector = selector.strip()
    if not selector:
        raise ValueError('Selector cannot be empty')
    if selector.startswith(('function:', 'class:')):
        return selector

    parts: List[Union[str, int]] = []
    token = ''
    i = 0
    while i < len(selector):
        ch = selector[i]
        if ch in '.':
            if token:
                parts.append(token)
                token = ''
            i += 1
            continue
        if ch == '[':
            if token:
                parts.append(token)
                token = ''
            end = selector.find(']', i)
            if end == -1:
                raise ValueError(f'Invalid selector "{selector}": missing "]"')
            idx = selector[i + 1:end]
            if not idx:
                raise ValueError(f'Invalid selector "{selector}": empty index')
            if idx.isdigit():
                parts.append(int(idx))
            else:
                parts.append(idx)
            i = end + 1
            continue
        token += ch
        i += 1
    if token:
        parts.append(token)
    return parts


def _unsupported_type_message(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    supported = supported_extensions()
    return (
        f'Unsupported file type "{ext or "<no extension>"}" for path "{path}". '
        f'Supported extensions: {", ".join(supported)}.'
    )


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
        handler = get_handler(path)
        if not handler:
            raise ValueError(_unsupported_type_message(path))
        outline = handler.outline(path, page, page_size)
        return {'kind': handler.kind, 'outline': outline}


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
                'description': (
                    'Tree: "function:<name>" or "class:<name>". '
                    'Map: ["a", 0, "b"] or "a[0].b". '
                    'Table: [row, col].'
                )
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
        selector = _parse_selector(params['selector'])
        page = params.get('page', 1)
        page_size = params.get('page_size', DEFAULT_PAGE_SIZE)
        handler = get_handler(path)
        if not handler:
            raise ValueError(_unsupported_type_message(path))
        result = handler.select(path, selector, page, page_size)
        if isinstance(result, dict):
            result = dict(result)
            result['kind'] = handler.kind
        return result


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
                'description': (
                    'Tree: "function:<name>" or "class:<name>". '
                    'Map: ["a", 0, "b"] or "a[0].b". '
                    'Table: [row, col].'
                )
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
        selector = _parse_selector(params['selector'])
        value = params['value']
        handler = get_handler(path)
        if not handler:
            raise ValueError(_unsupported_type_message(path))
        result = handler.replace(path, selector, value)
        if isinstance(result, dict) and 'kind' not in result:
            result['kind'] = handler.kind
        return result
