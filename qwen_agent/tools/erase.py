from typing import Dict, List, Optional, Union

from qwen_agent.llm.schema import ASSISTANT, FUNCTION, Message, USER, SYSTEM
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('erase')
class EraseTool(BaseTool):
    description = 'Erase messages from the current conversation context by index/range/role.'
    parameters = {
        'type': 'object',
        'properties': {
            'targets': {
                'type': 'array',
                'description': 'Selectors: index, range, or role+last.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'index': {'type': 'integer'},
                        'range': {
                            'type': 'object',
                            'properties': {
                                'start': {'type': 'integer'},
                                'end': {'type': 'integer'},
                            },
                        },
                        'role': {'type': 'string'},
                        'last': {'type': 'integer'},
                    },
                },
            },
            'reason': {'type': 'string'},
            'strategy': {'type': 'string', 'enum': ['summarize', 'drop']},
        },
        'required': ['targets', 'reason'],
    }

    def call(self, params: Union[str, dict], messages: Optional[List[Message]] = None, kernel=None, **kwargs) -> Dict:
        params = self._verify_json_format_args(params)
        if messages is None and kernel is not None:
            messages = kernel.working_context
        if messages is None:
            raise ValueError('erase requires current messages')

        targets = params.get('targets')
        reason = params.get('reason', '')
        strategy = params.get('strategy', 'summarize')
        if not targets:
            last_assistant = None
            if messages:
                for idx in range(len(messages) - 1, -1, -1):
                    if messages[idx].role == ASSISTANT:
                        last_assistant = idx
                        break
            example = None
            if last_assistant is not None:
                example = {
                    'targets': [{'index': last_assistant}],
                    'reason': 'Remove last assistant message to reduce context.',
                    'strategy': 'summarize',
                }

            def _snip(text: str, limit: int = 80) -> str:
                text = (text or '').replace('\n', ' ').strip()
                if len(text) <= limit:
                    return text
                return text[:limit].rstrip() + '...'

            preview = []
            if messages:
                start = max(0, len(messages) - 6)
                for idx in range(start, len(messages)):
                    msg = messages[idx]
                    content = msg.content
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                parts.append(item['text'])
                        content = ' '.join(parts)
                    preview.append({
                        'index': idx,
                        'role': msg.role,
                        'text': _snip(str(content)),
                    })

            return {
                'erased': [],
                'summary': 'No targets provided. Specify indices or ranges within the current conversation.',
                'help': {
                    'hint': 'Provide targets like {"index": 5} or {"range": {"start": 3, "end": 6}}.',
                    'example': example,
                    'recent': preview,
                },
            }

        erasable_roles = {ASSISTANT, FUNCTION, 'user'}
        to_erase = set()
        skipped = []
        skipped_reasons = []

        for target in targets:
            if 'index' in target:
                idx = target['index']
                if 0 <= idx < len(messages):
                    if messages[idx].role in erasable_roles:
                        to_erase.add(idx)
                    else:
                        skipped.append(idx)
                        skipped_reasons.append({'index': idx, 'role': messages[idx].role, 'reason': 'role_not_erasable'})
                continue
            if 'range' in target:
                start = target['range'].get('start', 0)
                end = target['range'].get('end', -1)
                if start > end:
                    start, end = end, start
                for idx in range(max(0, start), min(len(messages), end + 1)):
                    if messages[idx].role in erasable_roles:
                        to_erase.add(idx)
                    else:
                        skipped.append(idx)
                        skipped_reasons.append({'index': idx, 'role': messages[idx].role, 'reason': 'role_not_erasable'})
                continue
            if 'role' in target and 'last' in target:
                role = target['role']
                last = target['last']
                indices = [i for i, msg in enumerate(messages) if msg.role == role]
                for idx in indices[-last:]:
                    if messages[idx].role in erasable_roles:
                        to_erase.add(idx)
                continue

        if not to_erase:
            return {
                'erased': [],
                'skipped': skipped,
                'skipped_reasons': skipped_reasons,
                'summary': f'No erasable messages matched. Reason: {reason}',
            }

        kept = []
        erased_ids = []
        for i, msg in enumerate(messages):
            if i in to_erase:
                erased_ids.append(i)
            else:
                kept.append(msg)

        # Ensure valid turn order: first non-system message must be user.
        first_idx = None
        for idx, msg in enumerate(kept):
            if msg.role != SYSTEM:
                first_idx = idx
                break
        if first_idx is not None and kept[first_idx].role != USER:
            kept.insert(first_idx, Message(role=USER, content='[deleted]'))

        messages[:] = kept
        if kernel is not None:
            kernel.update_working_context(messages)
            kernel.mark_erased()

        summary = (
            f'Erased {len(erased_ids)} messages via {strategy}. '
            f'Reason: {reason}. '
            f'Erased indices: {erased_ids}'
        )
        return {
            'erased': erased_ids,
            'skipped': skipped,
            'skipped_reasons': skipped_reasons,
            'summary': summary,
        }
