import os
from typing import Any, Iterable


class ShapeHandler:
    kind: str = ''
    extensions: Iterable[str] = ()

    def can_handle(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in set(self.extensions)

    def outline(self, path: str, page: int, page_size: int) -> Any:
        raise NotImplementedError

    def select(self, path: str, selector: Any, page: int, page_size: int) -> Any:
        raise NotImplementedError

    def replace(self, path: str, selector: Any, value: Any) -> Any:
        raise NotImplementedError
