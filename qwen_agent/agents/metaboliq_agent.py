import copy
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.agents.assistant import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, Message
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

    def reset(self):
        self.working_context = []
        self.audit_journal = []
        self.erased_last_call = False
        self.public_history = []

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

    @staticmethod
    def _to_public_history(messages: List[Message]) -> List[Dict]:
        public = []
        for msg in messages:
            if msg.role in ('user', 'assistant'):
                public.append({'role': msg.role, 'content': msg.content})
        return public


SHAPE_TOOL_HINT = (
    'Tooling runs on the local filesystem. Absolute paths like "/Users/..." are valid inputs. '
    'Use describe_file/extract_section/replace_section directly on local paths when needed.'
)
ERASE_TOOL_HINT = (
    'The erase tool operates on the current conversation messages. '
    'It can remove assistant/tool messages by index or range.'
)


class MetaboliqAgent(Assistant):
    """Assistant with a lightweight kernel loop to manage working context."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 rag_cfg: Optional[Dict] = None,
                 kernel_cfg: Optional[Dict] = None):
        system_message = system_message or ''
        if SHAPE_TOOL_HINT not in system_message:
            if system_message:
                system_message = system_message + '\n\n' + SHAPE_TOOL_HINT
            else:
                system_message = SHAPE_TOOL_HINT
        if ERASE_TOOL_HINT not in system_message:
            system_message = system_message + '\n\n' + ERASE_TOOL_HINT
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
        if tool_name == 'erase':
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
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

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
                self.kernel.update_working_context(messages)
                used_any_tool = False
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        fn_msg = Message(role=FUNCTION,
                                         name=tool_name,
                                         content=tool_result,
                                         extra={'function_id': out.extra.get('function_id', '1')})
                        messages.append(fn_msg)
                        response.append(fn_msg)
                        self.kernel.update_working_context(messages)
                        yield response
                        used_any_tool = True
                if not used_any_tool:
                    break
        if response:
            self.kernel.record_turn(initial_messages, response)
        yield response
