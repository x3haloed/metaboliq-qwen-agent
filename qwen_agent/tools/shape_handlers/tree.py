import ast
from typing import Any, Dict

from qwen_agent.tools.shape_handlers.base import ShapeHandler
from qwen_agent.tools.shape_handlers.common import paginate_text, read_text, write_text


class TreeHandler(ShapeHandler):
    kind = 'tree'
    extensions = {'.py'}

    def outline(self, path: str, page: int, page_size: int) -> Dict[str, Any]:
        source = read_text(path)
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

    def select(self, path: str, selector: str, page: int, page_size: int) -> Dict[str, Any]:
        if not isinstance(selector, str):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        source = read_text(path)
        tree = ast.parse(source)
        kind, name = selector.split(':', 1)
        for node in tree.body:
            if kind == 'function' and isinstance(node, ast.FunctionDef) and node.name == name:
                segment = ast.get_source_segment(source, node) or ast.unparse(node)
                page_info = paginate_text(segment, page, page_size)
                return self._response(page_info)
            if kind == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
                segment = ast.get_source_segment(source, node) or ast.unparse(node)
                page_info = paginate_text(segment, page, page_size)
                return self._response(page_info)
        raise KeyError(f'{selector} not found')

    def replace(self, path: str, selector: str, value: Any) -> Dict[str, Any]:
        if not isinstance(selector, str):
            raise ValueError('Tree selector must be "function:<name>" or "class:<name>"')
        if not isinstance(value, str):
            raise ValueError('Tree replacement value must be source code')
        source = read_text(path)
        tree = ast.parse(source)
        kind, name = selector.split(':', 1)
        for node in tree.body:
            if kind == 'function' and isinstance(node, ast.FunctionDef) and node.name == name:
                self._replace_node(path, source, node, value)
                return {'changed': True, 'kind': self.kind}
            if kind == 'class' and isinstance(node, ast.ClassDef) and node.name == name:
                self._replace_node(path, source, node, value)
                return {'changed': True, 'kind': self.kind}
        raise KeyError(f'{selector} not found')

    @staticmethod
    def _replace_node(path: str, source: str, node: ast.AST, value: str) -> None:
        start = node.lineno - 1
        end = node.end_lineno or node.lineno
        lines = source.splitlines(keepends=True)
        replacement = value if value.endswith('\n') else value + '\n'
        write_text(path, ''.join(lines[:start]) + replacement + ''.join(lines[end:]))

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
