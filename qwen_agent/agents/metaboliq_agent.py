import copy
import json
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.agents.assistant import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import FUNCTION, ContentItem, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool


class KernelState:
    """In-memory kernel state for working context and audit events."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.working_context: List[Message] = []
        self.audit_journal: List[Dict] = []
        self.erased_last_call = False
        self.public_history: List[Dict] = []  # user/assistant only
        self.call_index = 0
        self.import_stage = 'idle'
        self.import_stage_started_at = 0
        self.import_stage_ttl_calls = int(self.cfg.get('import_stage_ttl_calls', 2))
        self.import_cap_chars = int(self.cfg.get('import_cap_chars', 1200))
        self.ephemeral_entries: List[Dict[str, Any]] = []
        self.promoted_message_ids = set()
        self.last_summary_message: Optional[Message] = None
        self.summary_requested = False

    def reset(self):
        self.working_context = []
        self.audit_journal = []
        self.erased_last_call = False
        self.public_history = []
        self.call_index = 0
        self.import_stage = 'idle'
        self.import_stage_started_at = 0
        self.ephemeral_entries = []
        self.promoted_message_ids = set()
        self.last_summary_message = None
        self.summary_requested = False

    def update_working_context(self, messages: List[Message]):
        self.working_context = messages
        self.public_history = self._to_public_history(messages)

    def record_turn(self, messages: List[Message], responses: List[Message]):
        self.audit_journal.append({
            'messages': [m.model_dump() for m in messages],
            'responses': [r.model_dump() for r in responses],
        })
        self.update_working_context(messages + responses)

    def mark_erased(self):
        self.erased_last_call = True

    def begin_llm_call(self, messages: List[Message]):
        self.call_index += 1
        self._enforce_stage_ttl(messages)
        self._mark_default_ephemeral(messages)
        self._prune_ephemeral(messages)

    def mark_ephemeral(self, msg: Message, ttl_calls: int = 1, kind: str = 'tool'):
        if id(msg) in self.promoted_message_ids:
            return
        self.ephemeral_entries.append({
            'message': msg,
            'expires_at': self.call_index + ttl_calls,
            'kind': kind,
        })

    def mark_summary_candidate(self, msg: Message):
        self.last_summary_message = msg
        self.mark_ephemeral(msg, ttl_calls=1, kind='summary')

    def promote_last_summary(self) -> Optional[Message]:
        if not self.last_summary_message:
            return None
        self._promote_message(self.last_summary_message)
        return self.last_summary_message

    def replace_with_summary(self, messages: List[Message], summary_msg: Message) -> None:
        keep_ids = {id(summary_msg)} | self.promoted_message_ids
        messages[:] = [m for m in messages if m.role in ('user', 'system') or id(m) in keep_ids]
        self._remove_ephemeral_except(keep_ids)
        self.last_summary_message = summary_msg

    def request_summary(self) -> None:
        self.summary_requested = True
        self.set_import_stage('summarize')

    def set_import_stage(self, stage: str) -> None:
        if stage != self.import_stage:
            self.import_stage = stage
            self.import_stage_started_at = self.call_index

    def _mark_default_ephemeral(self, messages: List[Message]) -> None:
        existing = {id(entry.get('message')) for entry in self.ephemeral_entries}
        for msg in messages:
            if msg.role in ('user', 'system'):
                continue
            if id(msg) in self.promoted_message_ids:
                continue
            if id(msg) in existing:
                continue
            self.mark_ephemeral(msg, ttl_calls=1, kind='auto')

    def _remove_ephemeral(self, msg: Message) -> None:
        self.ephemeral_entries = [
            entry for entry in self.ephemeral_entries if entry.get('message') is not msg
        ]

    def _remove_ephemeral_except(self, keep_ids: set) -> None:
        self.ephemeral_entries = [
            entry for entry in self.ephemeral_entries if id(entry.get('message')) in keep_ids
        ]

    def _promote_message(self, msg: Message) -> None:
        self.promoted_message_ids.add(id(msg))
        self._remove_ephemeral(msg)

    def _enforce_stage_ttl(self, messages: List[Message]) -> None:
        if self.import_stage == 'idle':
            return
        if self.import_stage_ttl_calls <= 0:
            return
        age = self.call_index - self.import_stage_started_at
        if age >= self.import_stage_ttl_calls:
            messages[:] = [m for m in messages if m.role in ('user', 'system')]
            self.ephemeral_entries = []
            self.last_summary_message = None
            self.promoted_message_ids = set()
            self.import_stage = 'idle'
            self.import_stage_started_at = self.call_index

    def _prune_ephemeral(self, messages: List[Message]):
        keep = []
        to_remove = set()
        for entry in self.ephemeral_entries:
            if entry.get('expires_at', 0) < self.call_index:
                msg = entry.get('message')
                if msg is not None:
                    to_remove.add(id(msg))
            else:
                keep.append(entry)
        if to_remove:
            messages[:] = [m for m in messages if id(m) not in to_remove]
            if self.last_summary_message and id(self.last_summary_message) in to_remove:
                self.last_summary_message = None
                if self.import_stage == 'summarize':
                    self.import_stage = 'idle'
                self.summary_requested = False
            notice = Message(
                role=FUNCTION,
                name='policy_notice',
                content={'expired_entries': len(to_remove), 'message': f'{len(to_remove)} entries expired'},
            )
            messages.append(notice)
            self.mark_ephemeral(notice, ttl_calls=1, kind='policy')
        self.ephemeral_entries = keep

    @staticmethod
    def _to_public_history(messages: List[Message]) -> List[Dict]:
        public = []
        for msg in messages:
            if msg.role in ('user', 'assistant'):
                public.append({'role': msg.role, 'content': msg.content})
        return public


SYSTEM_MESSAGE = (
    '''You operate under a finite, non-replayable working context.
If the context exceeds limits, execution halts.

Efficient operation requires selective attention, summarization, and erasure.
Intermediate reasoning and tool outputs are disposable unless explicitly preserved.

Use tools to inspect structure before ingesting content.
Prefer outlines, previews, and selectors over raw reads.

External storage is not working memory.
Reintegration is partial, costly, and must begin with summaries or outlines.
No single action may import large external state verbatim.

Erase low-value intermediate state proactively to preserve operational capacity.
Treat context pressure as a hard constraint, not a suggestion.

Proceed by:
   - scanning before loading
   - summarizing before retaining
   - pruning before continuing

Tooling runs on the local filesystem; absolute paths are valid.
The erase tool operates on the current conversation messages.'''
)


class MetaboliqAgent(Assistant):
    """Assistant with a lightweight kernel loop to manage working context."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None,
                 kernel_cfg: Optional[Dict] = None):
        system_message = SYSTEM_MESSAGE
        if function_list is None:
            function_list = []
        if not _has_tool(function_list, 'promote'):
            function_list = list(function_list) + ['promote']
        if not _has_tool(function_list, 'summarize'):
            function_list = list(function_list) + ['summarize']
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         rag_cfg=rag_cfg)
        self.kernel = KernelState(kernel_cfg)

    def reset_kernel(self):
        self.kernel.reset()

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs):
        if tool_name in {'erase', 'promote'}:
            kwargs['kernel'] = self.kernel
        return super()._call_tool(tool_name, tool_args, **kwargs)

    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'en',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        messages = self._prepend_knowledge_prompt(messages=messages, lang=lang, knowledge=knowledge, **kwargs)
        initial_messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        outline_tools = {'describe_file', 'storage.scan'}
        select_tools = {'extract_section', 'storage.get', 'retrieval'}
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            self.kernel.begin_llm_call(messages)
            self.kernel.update_working_context(messages)
            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output_stream = self._call_llm(messages=messages,
                                           functions=[func.function for func in self.function_map.values()],
                                           extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
                for out in output:
                    if out.role not in ('user', 'system'):
                        self.kernel.mark_ephemeral(out)
                self.kernel.update_working_context(messages)
                used_any_tool = False
                for out in output:
                    if out.extra is None:
                        out.extra = {}
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if out.role == 'assistant' and not use_tool:
                        if self.kernel.import_stage == 'summarize' and self.kernel.summary_requested:
                            out.content = _truncate_content(out.content, self.kernel.import_cap_chars)
                            self.kernel.replace_with_summary(messages, out)
                            self.kernel.mark_summary_candidate(out)
                            self.kernel.summary_requested = False
                    if use_tool:
                        parsed_args = _parse_tool_args(tool_args)
                        stage_tool = tool_name
                        if tool_name == 'storage' and isinstance(parsed_args, dict):
                            op = parsed_args.get('operate')
                            if op == 'scan':
                                stage_tool = 'storage.scan'
                            elif op == 'get':
                                stage_tool = 'storage.get'
                        if stage_tool in select_tools and self.kernel.import_stage != 'outline':
                            tool_result = {
                                'error': 'Reintegration requires outline → select → summarize → load.',
                                'detail': f'Select step "{stage_tool}" called before outline.',
                            }
                            fn_msg = Message(role=FUNCTION,
                                             name=stage_tool,
                                             content=tool_result,
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            response.append(fn_msg)
                            self.kernel.mark_ephemeral(fn_msg)
                            yield response
                            used_any_tool = True
                            continue
                        if tool_name == 'promote' and self.kernel.import_stage != 'summarize':
                            tool_result = {
                                'error': 'Reintegration requires outline → select → summarize → load.',
                                'detail': 'Promote called before summarize step.',
                            }
                            fn_msg = Message(role=FUNCTION,
                                             name=tool_name,
                                             content=tool_result,
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            response.append(fn_msg)
                            self.kernel.mark_ephemeral(fn_msg)
                            yield response
                            used_any_tool = True
                            continue
                        if tool_name == 'summarize' and self.kernel.import_stage != 'select':
                            tool_result = {
                                'error': 'Reintegration requires outline → select → summarize → load.',
                                'detail': 'Summarize called before select step.',
                            }
                            fn_msg = Message(role=FUNCTION,
                                             name=tool_name,
                                             content=tool_result,
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            response.append(fn_msg)
                            self.kernel.mark_ephemeral(fn_msg)
                            yield response
                            used_any_tool = True
                            continue
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        fn_content = tool_result
                        if tool_name == 'computer_use' and isinstance(tool_result, dict):
                            screenshot = tool_result.get('screenshot')
                            if screenshot:
                                safe_tool = dict(tool_result)
                                safe_tool.pop('screenshot', None)
                                fn_content = [
                                    ContentItem(image=screenshot),
                                    ContentItem(text=json.dumps(safe_tool, ensure_ascii=True)),
                                ]
                        if stage_tool in outline_tools:
                            self.kernel.set_import_stage('outline')
                        if stage_tool in select_tools:
                            self.kernel.set_import_stage('select')
                        if tool_name == 'summarize':
                            self.kernel.request_summary()
                        if tool_name == 'promote':
                            promoted = self.kernel.promote_last_summary()
                            if promoted:
                                self.kernel.set_import_stage('idle')
                        fn_msg = Message(role=FUNCTION,
                                         name=tool_name,
                                         content=_truncate_content(fn_content, self.kernel.import_cap_chars),
                                         extra={'function_id': out.extra.get('function_id', '1')})
                        messages.append(fn_msg)
                        response.append(fn_msg)
                        self.kernel.update_working_context(messages)
                        yield response
                        self.kernel.mark_ephemeral(fn_msg)
                        used_any_tool = True
                if not used_any_tool:
                    break
        if response:
            self.kernel.record_turn(initial_messages, response)
        yield response


def _has_promote(function_list: List[Union[str, Dict, BaseTool]]) -> bool:
    return _has_tool(function_list, 'promote')


def _has_tool(function_list: List[Union[str, Dict, BaseTool]], name: str) -> bool:
    for item in function_list:
        if item == name:
            return True
        if isinstance(item, dict) and item.get('name') == name:
            return True
        if isinstance(item, BaseTool) and getattr(item, 'name', None) == name:
            return True
    return False


def _parse_tool_args(tool_args: Union[str, dict]) -> Dict[str, Any]:
    if isinstance(tool_args, dict):
        return tool_args
    if not tool_args:
        return {}
    try:
        return json.loads(tool_args)
    except Exception:
        return {}


def _truncate_content(content: Union[str, List[ContentItem], dict], cap: int) -> Union[str, List[ContentItem]]:
    if cap <= 0:
        return content
    if isinstance(content, str):
        return content if len(content) <= cap else content[:cap] + '...'
    if isinstance(content, list):
        trimmed: List[ContentItem] = []
        remaining = cap
        for item in content:
            if item.text is not None:
                text = item.text
                if len(text) > remaining:
                    text = text[:remaining] + '...'
                    remaining = 0
                else:
                    remaining -= len(text)
                trimmed.append(ContentItem(text=text))
            else:
                trimmed.append(item)
        return trimmed
    try:
        serialized = json.dumps(content, ensure_ascii=True)
        if len(serialized) <= cap:
            return serialized
        return serialized[:cap] + '...'
    except Exception:
        return str(content)[:cap] + '...'
