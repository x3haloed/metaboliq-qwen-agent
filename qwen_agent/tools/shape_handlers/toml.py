from typing import Any

from qwen_agent.tools.shape_handlers.common import read_text, write_text
from qwen_agent.tools.shape_handlers.map import MapLikeHandler

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None

try:
    import tomli  # type: ignore
except ImportError:
    tomli = None

try:
    import tomli_w  # type: ignore
except ImportError:
    tomli_w = None

try:
    import toml  # type: ignore
except ImportError:
    toml = None


class TomlHandler(MapLikeHandler):
    kind = 'map'
    extensions = {'.toml'}

    def _load_map(self, path: str) -> Any:
        content = read_text(path)
        if tomllib is not None:
            return tomllib.loads(content)
        if tomli is not None:
            return tomli.loads(content)
        raise RuntimeError('toml support requires tomllib (py3.11+) or tomli')

    def _write_map(self, path: str, data: Any) -> None:
        if tomli_w is not None:
            write_text(path, tomli_w.dumps(data))
            return
        if toml is not None:
            write_text(path, toml.dumps(data))
            return
        raise RuntimeError('toml write support requires tomli-w or toml')
