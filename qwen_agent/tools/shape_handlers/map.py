import json
import os
from typing import Any, Dict, List

from qwen_agent.tools.shape_handlers.base import ShapeHandler
from qwen_agent.tools.shape_handlers.common import DEFAULT_MAX_CHARS, paginate_list, paginate_text, read_text, safe_json_size, write_text

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


class MapLikeHandler(ShapeHandler):
    kind = 'map'
    extensions = set()

    def outline(self, path: str, page: int, page_size: int) -> Dict[str, Any]:
        data = self._load_map(path)
        if isinstance(data, dict):
            keys = list(data.keys())
            page_info = paginate_list(keys, page, page_size)
            outline = {
                'summary': 'map',
                'keys': page_info['items'],
            }
            outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
            if page_info['truncated']:
                outline['note'] = (
                    f'Keys truncated. Call describe_file with page={page_info["next_page"]} '
                    f'to continue.'
                )
            return outline
        if isinstance(data, list):
            return {'summary': 'map-list', 'length': len(data)}
        return {'summary': 'map-scalar', 'type': type(data).__name__}

    def select(self, path: str, selector: List[Any], page: int, page_size: int) -> Dict[str, Any]:
        if not isinstance(selector, list):
            raise ValueError('Map selector must be a list path')
        data = self._load_map(path)
        obj = data
        for key in selector:
            try:
                obj = obj[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(f'No section found at {selector!r}: {exc}') from exc
        if isinstance(obj, dict):
            keys = list(obj.keys())
            if len(keys) <= page_size and safe_json_size(obj) <= DEFAULT_MAX_CHARS:
                return {'value': obj}
            page_info = paginate_list(keys, page, page_size)
            resp = self._paged_response(page_info)
            resp['value'] = page_info['items']
            if page_info['truncated']:
                resp['note'] = (
                    f'Keys truncated. Call extract_section with page={page_info["next_page"]} '
                    f'to continue, or select a deeper path.'
                )
            return resp
        if isinstance(obj, list):
            if len(obj) <= page_size and safe_json_size(obj) <= DEFAULT_MAX_CHARS:
                return {'value': obj}
            page_info = paginate_list(obj, page, page_size)
            resp = self._paged_response(page_info)
            resp['value'] = page_info['items']
            if page_info['truncated']:
                resp['note'] = (
                    f'List truncated. Call extract_section with page={page_info["next_page"]} '
                    f'to continue.'
                )
            return resp
        if isinstance(obj, str):
            page_info = paginate_text(obj, page, page_size)
            resp = self._paged_response(page_info)
            resp['value'] = page_info['text']
            if page_info['truncated']:
                resp['note'] = (
                    f'Text truncated. Call extract_section with page={page_info["next_page"]} '
                    f'to continue.'
                )
            return resp
        return {'value': obj}

    def replace(self, path: str, selector: List[Any], value: Any) -> Dict[str, Any]:
        if not isinstance(selector, list):
            raise ValueError('Map selector must be a list path')
        data = self._load_map(path)
        obj = data
        for key in selector[:-1]:
            try:
                obj = obj[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(f'No section found at {selector!r}: {exc}') from exc
        obj[selector[-1]] = value
        self._write_map(path, data)
        return {'changed': True, 'kind': self.kind}

    @staticmethod
    def _paged_response(page_info: Dict[str, Any]) -> Dict[str, Any]:
        return {k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')}

    @staticmethod
    def _load_map(self, path: str) -> Any:
        raise NotImplementedError

    def _write_map(self, path: str, data: Any) -> None:
        raise NotImplementedError


class MapHandler(MapLikeHandler):
    kind = 'map'
    extensions = {'.json', '.yaml', '.yml'}

    def _load_map(self, path: str) -> Any:
        ext = os.path.splitext(path)[1].lower()
        content = read_text(path)
        if ext == '.json':
            return json.loads(content)
        if ext in ('.yaml', '.yml'):
            if yaml is None:
                raise RuntimeError('yaml is required for .yaml/.yml support: pip install pyyaml')
            return yaml.safe_load(content)
        raise ValueError('Unsupported map format')

    def _write_map(self, path: str, data: Any) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.json':
            write_text(path, json.dumps(data, indent=2))
            return
        if ext in ('.yaml', '.yml'):
            if yaml is None:
                raise RuntimeError('yaml is required for .yaml/.yml support: pip install pyyaml')
            write_text(path, yaml.safe_dump(data, sort_keys=False))
            return
        raise ValueError('Unsupported map format')
