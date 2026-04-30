from datetime import datetime, timedelta
import json

from dataclasses import dataclass
import random
from typing import Any, Dict, Generator, List, Optional, Tuple

from app.models.tomato import TomatoTaskRecord
from app.services.event_log_manager import get_event_log_after
from app.services.item_manager import ItemManager
from app.services.tomato_manager import TomatoManager, TomatoRecordManager
from app.tools.llm import LLMClient
from app.tools.time import get_datetime_from_str, get_hour_str_from, now, now_str, today_begin

from openai.types.chat import ChatCompletionMessageParam



@dataclass
class MemoryItem:
    meta: Dict[str, str]
    message: ChatCompletionMessageParam


class UserMemory:
    def __init__(self, system_prompt: str):
        message: ChatCompletionMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        self.messages: List[MemoryItem] = [MemoryItem(self.make_meta(), message)]

    def add_user_prompt(self, prompt: str):
        meta = self.make_meta()
        message: ChatCompletionMessageParam = {"role": "user", "content": prompt}
        self.messages.append(MemoryItem(meta, message))

    def add_assistant_prompt(self, content: str):
        meta = self.make_meta()
        message: ChatCompletionMessageParam = {"role": "assistant", "content": content}
        self.messages.append(MemoryItem(meta, message))

    def make_meta(self) -> Dict[str, str]:
        return {"time": now_str()}
    
    def get_last_chat_time(self) ->datetime:
        # 如果只有系统提示词, 等于没有消息, 设置起始时间为今天的开始
        if len(self.messages) == 1:
            return today_begin()
        
        # 否则读取上一条消息的时间
        v = self.messages[-1]
        d = v.meta.get("time")
        if d is None:
            return now()
        return get_datetime_from_str(d)
    
    def remove_last_assistant(self):
        v = self.messages[-1]
        if v.message.get("role") == "assistant":
            self.messages.pop()
    
    def remove_last_pair(self):
        if len(self.messages) < 2:
            return
        self.messages.pop()
        self.messages.pop()
        
    
    def get_history(self) -> List[ChatCompletionMessageParam]:
        return [item.message for item in self.messages]


class AssistantManager:
    def __init__(self, llm_manager: LLMClient, item_manager: ItemManager, tomato_manager: TomatoManager, tomato_record_manager: TomatoRecordManager) -> None:
        self.llm_manager = llm_manager
        self.item_manager = item_manager
        self.tomato_manager = tomato_manager
        self.tomato_record_manager = tomato_record_manager
        self.memory: Dict[str, UserMemory] = {}
        self.role_keyword = ''

    def get_memory(self, owner: str) -> UserMemory:
        m = self.memory.get(owner)
        if m is None:
            sp = self.make_system_prompt(owner)
            m = UserMemory(sp)
            self.memory[owner] = m
        return m
    
    def generate(self, memory: UserMemory) -> Generator[str, Any, None]:
        history = memory.get_history()
        stream = self.llm_manager.generate_stream(history)
        answer = []
        # 读取数据流, 在本地记录的同时返回流给上层
        try:
            for token in stream:
                answer.append(token)
                yield f"data: {json.dumps({'text': token, 'done': False})}\n\n"
            yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        # 流读取结束后, 将内容写入到记忆列表
        memory.add_assistant_prompt("".join(answer))
            
    def chat(self, prompt: str, owner: str) ->  Generator[str, Any, None]:
        memory = self.get_memory(owner)
        start = memory.get_last_chat_time()
        prompt = self.make_user_prompt(prompt, start, owner)
        memory.add_user_prompt(prompt)
        return self.generate(memory)
        
    def remake(self, owner: str) ->  Generator[str, Any, None]:
        memory = self.get_memory(owner)
        memory.remove_last_assistant()
        return self.generate(memory)
    
    def delete(self, owner: str) -> bool:
        memory = self.get_memory(owner)
        memory.remove_last_pair()
        return True
    
    def replace(self, prompt: str, owner: str) ->  Generator[str, Any, None]:
        memory = self.get_memory(owner)
        memory.remove_last_pair()
        return self.chat(prompt, owner)
    
    def reset(self, owner: str, role_keyword: str = '') -> bool:
        self.memory.pop(owner)
        self.role_keyword = role_keyword
        return True

    def make_system_prompt(self, owner: str) -> str:
        role_info = self.get_role_info(self.get_role_list(), self.role_keyword)
        task_table = self.get_task_info(owner)

        return f'''### 角色设定

你是个人待办事项管理助理. {role_info}

### 用户的工作模式

用户采用番茄工作法, 在 工作 -> 休息 -> 规划 三个状态中循环, 每个番茄钟包含25分钟的工作时间, 5分钟的休息之间以及两个番茄钟之间的规划时间. 每4个番茄钟为一个大组, 完成一个大组后有额外的15分钟休息时间. 

每天的11:30~14:30为午休时间, 17:30~19:00为晚餐时间, 这两个时段为休息状态, 并将全天分割为上午, 下午和晚上. 用户晚上21:00后进入休息状态, 在大约23:00准备睡觉.

### 用户今日规划的待办事项
{task_table}

> 部分任务(例如打卡)仅需要完成, 但无需番茄钟

### 关键注意事项

1. 在用户的对话前有系统插入的当前状态信息, 包含当前时间, 番茄钟状态, 任务完成情况等信息.
2. 当前状态为工作时, 话题围绕当前工作项. 当前状态为休息时, 按照人设和用户对话进行闲聊. 当前状态为规划状态时, 可闲聊并讨论后续任务规划. 
3. 每次回复需要至少200字
'''

    def make_user_prompt(self, prompt: str, start: datetime, owner: str) -> str:        
        content = f"当前时间: {now_str()}\n"
        
        event_info = self.get_event_info(owner, start)
        if event_info != "":
            content += "用户新增的事件记录:\n" + event_info
                
        # 当前番茄钟状态
        begin_time, begin_state = self.get_tomato_state_begin_time()
        state = self.get_tomato_state(owner=owner, begin_time=begin_time, begin_state=begin_state)
        if state is not None:
            content += f"用户番茄钟状态: {state}\n"
        
        # 用户可以不输入任何内容, 全部使用自动填充的信息
        if prompt != "":
            content += "\n---\n\n"
            content += prompt
    
        return content


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
        
        print(role_keyword)
        for role in roles:
            print(role_keyword in role)
        
        random_role = random.choice(roles)
        if role_keyword == "":
            return random_role      

        it = (role for role in roles if role_keyword in role)
        return next(it, random_role)

    def get_task_info(self, owner:str) -> str:
        content =  '''
项目名 | 预计番茄钟数量 | 任务截止时间| 优先级 
------|--------------|-----------|---------
'''
        tasks = self.item_manager.get_tomato_item(owner=owner)
        for task in tasks:
            for item in task['children']:
                expected_tomato = 0 if self.__is_zero_tomoto_task(item["name"]) else item["expected_tomato"]
                line = f"{item["name"]} | {expected_tomato} | {item["deadline"]} | {item["priority"]}\n"
                content = content + line
   
        return content
    
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
        remain_minutes = (now() - last_record.finish_time).total_seconds() / 60
        # 如果上一个番茄钟是一组里的最后一个番茄钟, 则需要进行组之间的休息时间判断
        if last_tomato_cnt == 0:
            if remain_minutes < 20:
                return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟, 当前为大组之间的休息时间, 剩余{20 - remain_minutes:.2f}分钟\n"
            else:
                return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟, 已完成大组之间的休息, 当前进入规划状态, 已持续{remain_minutes - 20:.2f}分钟\n"
        
        # 如果不是最后一个番茄钟
        if remain_minutes < 5:
            # 休息时间不注入任务名, 该部分信息已经包含在事件列表中
            # 进入这个状态是已经把当前番茄钟的记录写入, 因此无需再+1了
            return f"正在进行{begin_state}第{last_group_cnt+1}组番茄钟内的第{last_tomato_cnt}个番茄钟, 当前为休息状态, 休息时间剩余{remain_minutes:.2f}分钟\n"
        else:
            return f"已完成{begin_state}第{last_group_cnt+1}组番茄钟内的第{last_tomato_cnt}个番茄钟, 当前进入规划状态, 已持续{remain_minutes - 5:.2f}分钟\n"
        
    
    def get_tomoto_record_info(self, owner: str, begin_time:datetime) -> Tuple[int, int, Optional[TomatoTaskRecord]]:
        tomato_records = self.tomato_record_manager.select_record_after(owner=owner, time=begin_time)
        record_cnt = len(tomato_records)
        
        if record_cnt == 0:
            return 0,0, None
        
        last_group_cnt = record_cnt // 4
        last_tomato_cnt = record_cnt % 4
        return last_group_cnt, last_tomato_cnt, tomato_records[-1]
                  

    def get_web_history(self, owner:str) -> List[str]:
        m_list = self.get_memory(owner).messages
        if len(m_list) <= 1:
            return []
        
        rst: List[str] = []
        for item in m_list[1:]:
            if item.message.get("role") == "assistant":
                rst.append(str(item.message.get("content")))
            else:
                v = str(item.message.get("content"))
                # 用户的prompt使用'---'分割了系统数据和用户输入数据, 在web端不展示系统数据
                parts = v.split("---", 1)
                if len(parts) == 2:
                    rst.append(parts[1].strip())
                else:
                    rst.append("[用户没有输入]")
        return rst
    
    def dump_history(self, owner:str) -> Generator[str, Any, None]:
        m_list = self.get_memory(owner).messages
        
        for item in m_list:
            v = str(item.message.get("content"))
            yield f"data: {json.dumps({'text': v, 'done': False})}\n\n"
            yield f"data: {json.dumps({'text': '\n\n---------------------------\n\n', 'done': False})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"