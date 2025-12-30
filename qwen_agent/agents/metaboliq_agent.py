import copy
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.agents.assistant import Assistant
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.tools import BaseTool


class KernelState:
    """In-memory kernel state for working context and audit events."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.working_context: List[Message] = []
        self.audit_journal: List[Dict] = []

    def reset(self):
        self.working_context = []
        self.audit_journal = []

    def record_turn(self, messages: List[Message], responses: List[Message]):
        self.audit_journal.append({
            'messages': [m.model_dump() for m in messages],
            'responses': [r.model_dump() for r in responses],
        })
        self.working_context = messages + responses


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

    def _run(self,
             messages: List[Message],
             lang: Literal['en', 'zh'] = 'en',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        messages_copy = copy.deepcopy(messages)
        last_responses: List[Message] = []
        for rsp in super()._run(messages=messages_copy, lang=lang, knowledge=knowledge, **kwargs):
            if rsp:
                last_responses = rsp
            yield rsp
        if last_responses:
            self.kernel.record_turn(messages_copy, last_responses)
