from typing import Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('promote')
class PromoteTool(BaseTool):
    description = 'Promote the last summary into working context (outline → select → summarize → load).'
    parameters = {
        'type': 'object',
        'properties': {
            'reason': {'type': 'string', 'description': 'Why this summary should be loaded.'},
            'target': {'type': 'string', 'description': 'Summary target to promote.', 'default': 'last_summary'},
        },
        'required': ['reason'],
    }

    def call(self, params: Union[str, dict], kernel=None, **kwargs) -> Dict:
        params = self._verify_json_format_args(params)
        if kernel is None:
            raise ValueError('promote requires kernel context')
        if kernel.import_stage != 'summarize':
            return {
                'promoted': False,
                'error': 'Promote requires outline → select → summarize → load.',
            }
        summary = kernel.last_summary_message
        if summary is None:
            return {'promoted': False, 'error': 'No summary candidate found.'}
        content = summary.content if isinstance(summary.content, str) else ''
        if kernel.import_cap_chars and len(content) > kernel.import_cap_chars:
            return {
                'promoted': False,
                'error': f'Summary exceeds import cap of {kernel.import_cap_chars} chars.',
            }
        return {
            'promoted': True,
            'summary_preview': content[:200],
        }
