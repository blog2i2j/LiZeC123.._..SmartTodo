from datetime import datetime, timedelta
import json
import random
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionToolUnionParam
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.chat.chat_completion_function_tool_param import ChatCompletionFunctionToolParam
import sqlalchemy as sal
from sqlalchemy.orm import Session, scoped_session

from app.models.assistant import AssistantHistory, AssistantType
from app.models.item import Item
from app.models.tomato import TomatoTaskRecord
from app.services.event_log_manager import get_event_log_after
from app.services.item_manager import ItemManager
from app.services.tomato_manager import TomatoManager, TomatoRecordManager
from app.tools.llm import LLMClient
from app.tools.log import logger
from app.tools.time import get_datetime_from_str, get_hour_str_from, now, parse_deadline_timestamp, today_begin




# 系统提示词仅包含固定内容
SystemPrompt = ChatCompletionSystemMessageParam(
    role="system",
    content='''你是用户的个人待办事项管理助理.

### 用户的工作模式

用户采用番茄工作法, 在 工作 -> 休息 -> 规划 三个状态中循环, 每个番茄钟包含25分钟的工作时间, 5分钟的休息时间. 两个番茄钟之间属于规划时间. 每4个番茄钟为一个大组, 完成一个大组后有额外的15分钟休息时间. 

每天的11:30~14:30为午休时间, 17:30~19:00为晚餐时间, 这两个时段为休息状态, 并将全天分割为上午, 下午和晚上. 用户晚上21:00后进入休息状态, 在大约23:00准备睡觉.

### 系统权限

你具有创建新的待办事项的权限, 你可以调用`create_item`工具进行处理. 事项名称的格式为: [助手名称]:[事项主题]

你既可以创建用户明确要求的事项, 也可以创建聊天过程中自然提及的其他事项. 但对于不是用户明确要求的事项需要判断:
1. 这个事件涉及未来某个具体时间点或时间段
2. 这个事件目前不在待办系统中
3. 这个事件对用户或者助手很重要, 忘记的话会造成麻烦

创建前需要逐一检查是否满足这三个要求, 然后再进行创建.

### 关键注意事项

1. 在用户的对话前有系统插入的当前状态信息,和用户行为日志.
2. 当前状态为工作时, 话题围绕当前工作项. 当前状态为休息时, 按照人设和用户对话进行闲聊. 当前状态为规划状态时, 可闲聊并讨论后续任务规划. 
3. 每次回复需要至少200字
'''
)


# 创建待办事项工具
CreatItemTool: ChatCompletionFunctionToolParam = ChatCompletionFunctionToolParam(
    type="function",
    function=FunctionDefinition(
        name="create_item",
        description="创建一个新的待办事项",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "待办事项的名称",
                },
                "deadline": {
                    "type": "string",
                    "format": "date-time",
                    "description": "截止时间, 格式为2006-01-02 15:04:05",
                },
                "priority": {
                    "type": "string",
                    "description": "优先级，常用取值：'p0'（最高）、'p1'（默认）、'p2'",
                    "default": "p1",
                    "enum": ["p0", "p1", "p2"],
                },
            },
            "required": ["name", "deadline", "priority"],
        },
    ),
)

class AssistantHistoryManager:
    def __init__(self, db: scoped_session[Session]) -> None:
        self.db = db
        pass

    def add_user_prompt(self, prompt: str, inject: str, owner:str):
        msg = AssistantHistory(role='user', content=prompt, system_inject_content=inject, owner=owner)
        self.db.add(msg)
        self.db.flush()
        self.db.commit()
    
    def add_assistant_prompt(self, content: str,owner:str):
        msg = AssistantHistory(role='assistant', content=content, owner=owner)
        self.db.add(msg)
        self.db.flush()
        self.db.commit()
        
    def remove_last_assistant(self, owner: str) -> bool:
        last =  self.select_last_msg(owner)
        if last is None:
            return False
        
        if last.role != AssistantType.Assistant:
            return False
        
        self.db.delete(last)
        self.db.flush()
        self.db.commit()
        return True
    
    def remove_last_user(self, owner: str) -> bool:
        last =  self.select_last_msg(owner)
        if last is None:
            return False
        
        if last.role != AssistantType.User:
            return False
        
        self.db.delete(last)
        self.db.flush()
        self.db.commit()
        return True
    
    def remove_last_pair(self, owner: str) -> bool:
        a = self.remove_last_assistant(owner)
        u = self.remove_last_user(owner)
        return a and u

    def get_last_chat_time(self, owner: str) ->datetime:
        """获取上一条消息的时间, 如果该用户没有任何消息, 则上一条消息的时间视为今天的开始时间"""
        last = self.select_last_msg(owner)
        if last is None:
            return today_begin()
        else:
            return last.create_time        
    
    def select_last_msg(self, owner: str) -> Optional[AssistantHistory]:
        stmt = sal.select(AssistantHistory).where(AssistantHistory.owner==owner).order_by(AssistantHistory.id.desc()).limit(1)
        return self.db.scalar(stmt)
    
    def select_record(self, owner:str, start_time: datetime) -> Iterable[AssistantHistory]:
        stmt = sal.select(AssistantHistory).where(AssistantHistory.owner == owner, AssistantHistory.create_time > start_time)
        return self.db.scalars(stmt)

    def get_history(self, owner, start_time: datetime)-> List[ChatCompletionMessageParam]:
        records = self.select_record(owner, start_time)
        return [SystemPrompt] + [msg.to_openai() for msg in records]
    

class AssistantManager:
    def __init__(self, llm_manager: LLMClient, item_manager: ItemManager, 
                 tomato_manager: TomatoManager, tomato_record_manager: TomatoRecordManager,
                 history_manager: AssistantHistoryManager) -> None:
        self.llm_manager = llm_manager
        self.item_manager = item_manager
        self.tomato_manager = tomato_manager
        self.tomato_record_manager = tomato_record_manager
        self.history_manager = history_manager
        self.start_time : Dict[str, datetime] = {}
    
    def generate(self, owner: str, /, enable_tools=False) -> Generator[str, Any, None]:
        """流式生成回复：后台消费 LLM 流并保存，前台推送给客户端"""
        start_time = self.get_start_time(owner)
        history = self.history_manager.get_history(owner, start_time)
        if enable_tools:
            print("进入包含工具的分支")
            tool_desc, tool_map = self.make_tools(owner)
            stream = self.llm_manager.generate_steam_with_tools(history, tool_desc, tool_map)
        else:
            stream = self.llm_manager.generate_stream(history)
        full_answer = []
        
        try:
            for token in stream:
                full_answer.append(token)
                yield f"data: {json.dumps({'text': token, 'done': False})}\n\n"
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        except GeneratorExit as e:
            logger.error(f'推送LLM模型消息到客户端中断: {e}')
            # 继续消费上游数据, 确保已经生成的内容依然可以落库
            for token in stream:
                full_answer.append(token)
        except Exception as e:
            # 发生其他异常时，先尝试发送错误信息（如果客户端还在）
            logger.error(f'推送LLM模型消息到客户端异常: {e}')
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n" 
        finally:
            content = "".join(full_answer)
            self.history_manager.add_assistant_prompt(content, owner)
    
    def get_start_time(self, owner: str) -> datetime:
        if owner not in self.start_time:
            self.start_time[owner] = today_begin() - timedelta(days=2)
        
        return self.start_time[owner]

    def make_tools(self, owner:str) -> Tuple[Iterable[ChatCompletionToolUnionParam], Dict[str, Callable[[str], str]]]: 
        def create_f(arg_json:str) -> str:
            try:
                print(f"执行创建事项函数: {arg_json}")
                args:Dict[str,str] = json.loads(arg_json)
                item = Item(name=args.get('name'), item_type='single', owner=owner,
                            deadline=get_datetime_from_str(args.get('deadline', '')),
                            priority=args.get('priority'))
                self.item_manager.create(item)
                self.item_manager.db.commit()
            except Exception as e:
                return f"error: {e}"
            return "success"

        
        return [CreatItemTool], {"create_item": create_f}
        
             
                
    def chat(self, prompt: str, owner: str) ->  Generator[str, Any, None]:
        start = self.history_manager.get_last_chat_time(owner)
        inject_content = self.make_user_inject_content(start, owner)
        self.history_manager.add_user_prompt(prompt, inject_content, owner)
        return self.generate(owner, enable_tools=True) # TODO: 使用开关控制
        
    def remake(self, owner: str) ->  Generator[str, Any, None]:
        self.history_manager.remove_last_assistant(owner)
        return self.generate(owner)
    
    def delete(self, owner: str) -> bool:
        return self.history_manager.remove_last_pair(owner)
        
    def replace(self, prompt: str, owner: str) ->  Generator[str, Any, None]:
        self.history_manager.remove_last_pair(owner)
        return self.chat(prompt, owner)
    
    def reset(self, owner: str, role_keyword: str = '') -> Generator[str, Any, None]:
        self.start_time[owner] = now()
        role_info = self.make_replace_role_prompt(role_keyword)
        inject = self.make_user_inject_content(today_begin(), owner) # 操作信息要从今天开始的时间读取
        content = inject  + "\n\n---\n\n" + role_info
        self.history_manager.add_user_prompt("", content, owner)
        
        return self.generate(owner)
    
    def replace_role(self, keyword: str, prompt:str, owner: str) -> Generator[str, Any, None]:
        inject = self.make_replace_role_prompt(keyword)
        self.history_manager.add_user_prompt(prompt, inject, owner)
        return self.generate(owner)
    
    
    def make_user_inject_content(self, start: datetime, owner:str) -> str:
        # 当前番茄钟状态
        begin_time, begin_state = self.get_tomato_state_begin_time()
        state = self.get_tomato_state(owner=owner, begin_time=begin_time, begin_state=begin_state)
        content = f"番茄钟状态: {state}\n"
        
        # 事件信息, 可能没有事件
        event_info = self.get_event_info(owner, start)
        if event_info != "":
            content += "用户新增的事件记录:\n" + event_info
            
        return content


    def make_replace_role_prompt(self, keyword: str) -> str:
        role_info = self.get_role_info(self.get_role_list(), keyword)
        return f"角色切换: {role_info}. 现在由你接替之前的个人助理"

    def get_role_list(self) -> List[str]:
        try:
            with open("config/role/Assistant.md") as f:
                return [role.strip() for role in f if role.strip() != ""]
        except OSError:
            # 文件不存在时, 直接返回空即可, 相当于没有额外的角色设定
            return []
    
    def get_role_info(self, roles: List[str], role_keyword: str) -> str:
        if len(roles) == 0:
            return ""
        
        random_role = random.choice(roles)
        if role_keyword == "":
            return random_role      

        it = (role for role in roles if role_keyword in role)
        return next(it, random_role)
    
    def __is_zero_tomoto_task(self, name:str) -> bool:
        # 打卡类任务可瞬间完成无需番茄钟.  午间和晚间任务不占用番茄钟
        keywords = ['打卡', '午间', '晚间']
        return any(word in name for word in keywords)
    
    
    def get_event_info(self, owner:str, begin_time: datetime) -> str:
        content = ""
        # 新增番茄钟记录
        events = get_event_log_after(self.item_manager.db, begin_time, owner)
        for e in events:
            content += f"{get_hour_str_from(e.create_time)}: {e.msg}\n"        
        
        return content
    
    def get_tomato_state_begin_time(self) -> Tuple[datetime, str]:
        now_time = now()
        today_morning_start = datetime(now_time.year, now_time.month, now_time.day, 8, 0, 0) 
        today_morning_end = datetime(now_time.year, now_time.month, now_time.day, 12, 0, 0)
        today_afternoon_end = datetime(now_time.year, now_time.month, now_time.day, 18, 0, 0)
        
        if now_time < today_morning_end:
            return today_morning_start, '上午'
        
        if now_time < today_afternoon_end:
            return today_morning_end, '下午'
        
        return today_afternoon_end, '晚上'
    
    def check_rest_time(self) -> str:
        now_time = now()
        noon_rest_start = datetime(now_time.year, now_time.month, now_time.day, 11, 30, 0) 
        noon_rest_end = datetime(now_time.year, now_time.month, now_time.day, 14, 30, 0) 
        
        evening_start = datetime(now_time.year, now_time.month, now_time.day, 17, 30, 0) 
        evening_end = datetime(now_time.year, now_time.month, now_time.day, 19, 00, 0) 
        
        night_start = datetime(now_time.year, now_time.month, now_time.day, 21, 00, 0)
        
        if noon_rest_start < now_time < noon_rest_end:
            return "午间休息时间"
        
        if evening_start < now_time < evening_end:
            return "晚间休息时间"
        
        if now_time > night_start:
            return "深夜休息时间"
        
        return ""

    
    def get_tomato_state(self, owner: str, begin_time: datetime, begin_state: str) -> str:
        # 首先检查是否是番茄钟工作状态, 该状态优先级最高, 因此用户实际上可以在任意时间开始番茄钟
        state = self.tomato_manager.query_task(owner=owner)
        if state:
            last_group_cnt, last_tomato_cnt, _ = self.get_tomoto_record_info(owner=owner, begin_time=begin_time)
            remain_minutes = (state.start_time + timedelta(minutes=25) - now()).total_seconds() / 60
            return f"正在进行{begin_state}第{last_group_cnt+1}组番茄钟内的第{last_tomato_cnt+1}个番茄钟, 当前为工作状态, 工作项目为[{state.name}], 工作时间剩余{remain_minutes:.2f}分钟\n"

        # 其次检查是否为休息时间, 相当于可以覆盖番茄钟的休息和规划状态
        reset_time = self.check_rest_time()
        if reset_time != "":
            return reset_time
        
        # 当前不是番茄钟状态, 先检查是否为初始状态
        last_group_cnt, last_tomato_cnt, last_record = self.get_tomoto_record_info(owner=owner, begin_time=begin_time)
        if last_record is None:
            # 没有开始任何番茄钟
            return f"还未开始任何番茄钟\n"
        
        # 不是初始状态, 再检查休息和规划状态
        elapsed_minutes = (now() - last_record.finish_time).total_seconds() / 60
        # 如果上一个番茄钟是一组里的最后一个番茄钟, 则需要进行组之间的休息时间判断
        if last_tomato_cnt == 0:
            if elapsed_minutes < 20:
                return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟, 当前为大组之间的休息时间, 剩余{20 - elapsed_minutes:.2f}分钟\n"
            else:
                return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟, 已完成大组之间的休息, 当前进入规划状态, 已持续{elapsed_minutes - 20:.2f}分钟\n"
        
        # 如果不是最后一个番茄钟
        if elapsed_minutes < 5:
            # 休息时间不注入任务名, 该部分信息已经包含在事件列表中
            # 进入这个状态是已经把当前番茄钟的记录写入, 因此无需再+1了
            return f"正在进行{begin_state}第{last_group_cnt+1}组番茄钟内的第{last_tomato_cnt}个番茄钟, 当前为休息状态, 休息时间剩余{5 - elapsed_minutes:.2f}分钟\n"
        else:
            return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟内的第{last_tomato_cnt}个番茄钟, 当前进入规划状态, 已持续{elapsed_minutes - 5:.2f}分钟\n"
        
    
    def get_tomoto_record_info(self, owner: str, begin_time:datetime) -> Tuple[int, int, Optional[TomatoTaskRecord]]:
        tomato_records = self.tomato_record_manager.select_record_after(owner=owner, time=begin_time)
        record_cnt = len(tomato_records)
        
        if record_cnt == 0:
            return 0,0, None
        
        last_group_cnt = record_cnt // 4
        last_tomato_cnt = record_cnt % 4
        return last_group_cnt, last_tomato_cnt, tomato_records[-1]
                  

    def get_web_history(self, owner:str) -> List[str]:
        start_time = self.get_start_time(owner)
        record = self.history_manager.select_record(owner, start_time)
        msg = [msg.to_web() for msg in record]
        return [m for m in msg if m is not None]
        
    
    def dump_history(self, owner:str) -> Generator[str, Any, None]:
        start_time = self.get_start_time(owner)
        record = self.history_manager.select_record(owner, start_time)
        
        for item in record:
            v = item.to_openai()
            v = str(f"{v.get('role')}: {v.get("content")}\n\n")
            yield f"data: {json.dumps({'text': v, 'done': False})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        
    def dump_user_prompt(self, owner: str) ->Generator[str, Any, None]:
        start = self.history_manager.get_last_chat_time(owner)
        content = self.make_user_inject_content(start, owner)
        yield f"data: {json.dumps({'text': content, 'done': False})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"