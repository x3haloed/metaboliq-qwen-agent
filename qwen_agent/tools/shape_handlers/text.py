from typing import Any, Dict

from qwen_agent.tools.shape_handlers.base import ShapeHandler
from qwen_agent.tools.shape_handlers.common import paginate_text, paginate_list, read_text


class TextHandler(ShapeHandler):
    kind = 'text'
    extensions = {'.txt', '.log', '.md', '.markdown'}

    def outline(self, path: str, page: int, page_size: int) -> Dict[str, Any]:
        text = read_text(path)
        ext = path.lower().split('.')[-1]
        if ext in ('md', 'markdown'):
            return self._outline_markdown(text, page, page_size)
        page_info = paginate_text(text, page, page_size)
        outline = {
            'summary': 'text',
            'preview': page_info['text'],
        }
        outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
        if page_info['truncated']:
            outline['note'] = (
                f'Text truncated. Call describe_file with page={page_info["next_page"]} '
                f'to continue.'
            )
        return outline

    def select(self, path: str, selector: Any, page: int, page_size: int) -> Dict[str, Any]:
        text = read_text(path)
        page_info = paginate_text(text, page, page_size)
        resp = {
            'value': page_info['text'],
        }
        resp.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
        if page_info['truncated']:
            resp['note'] = (
                f'Text truncated. Call extract_section with page={page_info["next_page"]} '
                f'to continue.'
            )
        return resp

    def replace(self, path: str, selector: Any, value: Any) -> Dict[str, Any]:
        raise ValueError('replace not supported for text')

    @staticmethod
    def _outline_markdown(text: str, page: int, page_size: int) -> Dict[str, Any]:
        lines = text.splitlines()
        headings = []
        for line in lines:
            if line.lstrip().startswith('#'):
                headings.append(line.strip())
        page_info = paginate_list(headings, page, page_size)
        outline = {
            'summary': 'markdown',
            'headings': page_info['items'],
        }
        outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
        if page_info['truncated']:
            outline['note'] = (
                f'Headings truncated. Call describe_file with page={page_info["next_page"]} '
                f'to continue.'
            )
        return outline
