import re
from typing import Any, Dict, List, Tuple

from qwen_agent.tools.shape_handlers.base import ShapeHandler
from qwen_agent.tools.shape_handlers.common import paginate_text, read_text, write_text


class JsTsHandler(ShapeHandler):
    kind = 'tree'
    extensions = {'.js', '.ts', '.jsx', '.tsx'}

    _fn_decl = re.compile(r'^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', re.MULTILINE)
    _class_decl = re.compile(r'^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b', re.MULTILINE)
    _arrow_decl = re.compile(
        r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
        re.MULTILINE,
    )

    def outline(self, path: str, page: int, page_size: int) -> Dict[str, Any]:
        source = read_text(path)
        functions = sorted(set(self._fn_decl.findall(source) + self._arrow_decl.findall(source)))
        classes = sorted(set(self._class_decl.findall(source)))
        return {
            'summary': 'tree',
            'functions': functions,
            'classes': classes,
        }

    def select(self, path: str, selector: str, page: int, page_size: int) -> Dict[str, Any]:
        if not isinstance(selector, str):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        source = read_text(path)
        kind, name = selector.split(':', 1)
        if kind not in ('function', 'class'):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        start, end = self._find_block(source, kind, name)
        snippet = source[start:end]
        page_info = paginate_text(snippet, page, page_size)
        return self._response(page_info)

    def replace(self, path: str, selector: str, value: Any) -> Dict[str, Any]:
        if not isinstance(selector, str):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        if not isinstance(value, str):
            raise ValueError('Tree replacement value must be source code')
        source = read_text(path)
        kind, name = selector.split(':', 1)
        if kind not in ('function', 'class'):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        start, end = self._find_block(source, kind, name)
        replacement = value if value.endswith('\n') else value + '\n'
        write_text(path, source[:start] + replacement + source[end:])
        return {'changed': True, 'kind': self.kind}

    def _find_block(self, source: str, kind: str, name: str) -> Tuple[int, int]:
        if kind == 'function':
            pattern = re.compile(rf'^\s*function\s+{re.escape(name)}\s*\(', re.MULTILINE)
        else:
            pattern = re.compile(rf'^\s*class\s+{re.escape(name)}\b', re.MULTILINE)
        match = pattern.search(source)
        if not match:
            raise KeyError(f'{kind}:{name} not found')
        start = match.start()
        brace_start = source.find('{', match.end())
        if brace_start == -1:
            line_end = source.find('\n', match.end())
            return start, len(source) if line_end == -1 else line_end + 1
        end = self._match_brace(source, brace_start)
        return start, end

    @staticmethod
    def _match_brace(source: str, brace_start: int) -> int:
        depth = 0
        i = brace_start
        while i < len(source):
            ch = source[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(source)

    @staticmethod
    def _response(page_info: Dict[str, Any]) -> Dict[str, Any]:
        resp = {
            'value': page_info['text'],
            'page': page_info['page'],
            'page_size': page_info['page_size'],
            'total': page_info['total'],
            'truncated': page_info['truncated'],
            'next_page': page_info['next_page'],
        }
        if page_info['truncated']:
            resp['note'] = (
                f'Text truncated. Call extract_section with page={page_info["next_page"]} '
                f'to continue.'
            )
        return resp
