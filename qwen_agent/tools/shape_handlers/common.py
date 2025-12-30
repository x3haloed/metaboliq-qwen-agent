import json
from typing import Any, Dict, List

DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_CHARS = 4000


def read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def safe_json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=True))
    except Exception:
        return len(str(value))


def paginate_list(items: List[Any], page: int, page_size: int) -> Dict[str, Any]:
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


def paginate_text(text: str, page: int, page_size: int) -> Dict[str, Any]:
    lines = text.splitlines()
    page_info = paginate_list(lines, page, page_size)
    page_text = '\n'.join(page_info['items'])
    return {
        'text': page_text,
        'page': page_info['page'],
        'page_size': page_info['page_size'],
        'total': page_info['total'],
        'truncated': page_info['truncated'],
        'next_page': page_info['next_page'],
    }
