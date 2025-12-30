import json
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import add_qwen_libs  # NOQA
except ImportError:
    pass

from qwen_agent.agents import MetaboliqAgent
from qwen_agent.gui.web_ui import WebUI
from qwen_server.schema import GlobalConfig


if load_dotenv:
    load_dotenv()

server_config_path = Path(__file__).resolve().parent / 'server_config.json'
with open(server_config_path, 'r') as f:
    server_config = json.load(f)
    server_config = GlobalConfig(**server_config)

llm_config = None
if hasattr(server_config.server, 'llm'):
    llm_config = {
        'model': server_config.server.llm,
        'api_key': server_config.server.api_key,
        'model_server': server_config.server.model_server,
    }
    if getattr(server_config.server, 'model_type', ''):
        llm_config['model_type'] = server_config.server.model_type

agent = MetaboliqAgent(
    llm=llm_config,
    name='metaboliq',
    description='Token-metabolism agent with multimodal tool use.',
)

ui = WebUI(
    agent,
    chatbot_config={
        'input.placeholder': 'Ask Metaboliq...',
    },
)

ui.run(
    server_name=server_config.server.server_host,
    server_port=server_config.server.workstation_port,
)
