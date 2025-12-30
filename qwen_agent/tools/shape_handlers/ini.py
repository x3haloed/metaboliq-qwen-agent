import configparser
from typing import Any, Dict

from qwen_agent.tools.shape_handlers.map import MapLikeHandler


class IniHandler(MapLikeHandler):
    kind = 'map'
    extensions = {'.ini', '.cfg'}

    def _load_map(self, path: str) -> Dict[str, Any]:
        parser = configparser.ConfigParser()
        parser.read(path)
        data: Dict[str, Any] = {}
        if parser.defaults():
            data['DEFAULT'] = dict(parser.defaults())
        for section in parser.sections():
            data[section] = dict(parser.items(section))
        return data

    def _write_map(self, path: str, data: Dict[str, Any]) -> None:
        parser = configparser.ConfigParser()
        defaults = data.get('DEFAULT')
        if isinstance(defaults, dict):
            parser['DEFAULT'] = {str(k): str(v) for k, v in defaults.items()}
        for key, value in data.items():
            if key == 'DEFAULT':
                continue
            if isinstance(value, dict):
                parser[key] = {str(k): str(v) for k, v in value.items()}
            else:
                parser[key] = {'value': str(value)}
        with open(path, 'w', encoding='utf-8') as f:
            parser.write(f)
