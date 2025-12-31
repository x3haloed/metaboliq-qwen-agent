from typing import Dict, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('summarize')
class SummarizeTool(BaseTool):
    description = 'Request a summarize step (outline → select → summarize → load).'
    parameters = {
        'type': 'object',
        'properties': {
            'reason': {'type': 'string', 'description': 'Why a summary is needed.'},
        },
        'required': ['reason'],
    }

    def call(self, params: Union[str, dict], **kwargs) -> Dict:
        self._verify_json_format_args(params)
        return {'requested': True}
