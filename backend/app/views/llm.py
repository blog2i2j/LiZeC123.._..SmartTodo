from typing import Dict

from flask import Blueprint, request, Response

from app import assistant_manager
from app.views.authority import authority_check

llm_bp = Blueprint('llm', __name__)


@llm_bp.post('/api/stream/assistant/chat')
@authority_check()
def assistant_chat_stream(owner: str):
    f: Dict = request.get_json()
    prompt: str = f.get('prompt', '')
    
    if prompt == '/du':
        # display user inject prompt
        g = assistant_manager.dump_user_prompt(owner)
    elif prompt == '/da':
        # display all
        g = assistant_manager.dump_history(owner)
    elif prompt == '/rk':
        # remake answer
        g = assistant_manager.remake(owner)
    elif prompt.startswith("/rr "):
        args = [arg for arg in prompt.strip().split() if arg]
        if len(args) >= 3:
            # 指定的角色并给出了prompt
            g = assistant_manager.replace_role(args[1],args[2], owner)
        elif len(args) == 2:
            # 仅给出角色, 无prompt
            g = assistant_manager.replace_role(args[1], '', owner)
        else:
            # 没有任何附加信息, 随机切换一个角色
            g = assistant_manager.replace_role('', '', owner)
    elif prompt.startswith("/rc "):
        # replace content
        args = [arg for arg in prompt.strip().split() if arg]
        g = assistant_manager.replace(args[1], owner)
    elif prompt.startswith("/rs"):
        # reset 
        args = [arg for arg in prompt.strip().split() if arg]
        if len(args) >= 2:
            g = assistant_manager.reset(owner, args[1])
        else:
            g = assistant_manager.reset(owner)
    else:
        g = assistant_manager.chat(prompt, owner)

    return Response(g, mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
        }
    )


@llm_bp.post('/api/assistant/history')
@authority_check()
def assistant_history(owner: str):
    return assistant_manager.get_web_history(owner)


@llm_bp.post('/api/assistant/delete')
@authority_check()
def assistant_delete(owner: str):
    return assistant_manager.delete(owner)
