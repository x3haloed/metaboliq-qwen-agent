from qwen_agent.tools.shape_handlers.ini import IniHandler
from qwen_agent.tools.shape_handlers.js_ts import JsTsHandler
from qwen_agent.tools.shape_handlers.map import MapHandler
from qwen_agent.tools.shape_handlers.table import TableHandler
from qwen_agent.tools.shape_handlers.text import TextHandler
from qwen_agent.tools.shape_handlers.toml import TomlHandler
from qwen_agent.tools.shape_handlers.tree import TreeHandler

HANDLERS = [
    TreeHandler(),
    MapHandler(),
    TableHandler(),
    TextHandler(),
    JsTsHandler(),
    TomlHandler(),
    IniHandler(),
]


def get_handler(path: str):
    for handler in HANDLERS:
        if handler.can_handle(path):
            return handler
    return None


def supported_extensions() -> list:
    exts = set()
    for handler in HANDLERS:
        exts.update(handler.extensions)
    return sorted(exts)
