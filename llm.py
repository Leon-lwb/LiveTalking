import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        from openai import OpenAI
        llm_provider = getattr(opt, 'llm', 'dashscope')
        if llm_provider == 'orcarouter':
            # OrcaRouter: 兼容 OpenAI 的 LLM 路由网关
            client = OpenAI(
                api_key=os.getenv("ORCAROUTER_API_KEY"),
                base_url="https://api.orcarouter.ai/v1",
            )
            llm_model = getattr(opt, 'llm_model', None) or 'orcarouter/auto'
        else:
            # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                # 填写DashScope SDK的base_url
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            llm_model = getattr(opt, 'llm_model', None) or 'qwen-plus'
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{'role': 'system', 'content': '你是一个知识助手，尽量以简短、口语化的方式输出'},
                    {'role': 'user', 'content': message}],
            stream=True,
            # 通过以下设置，在流式输出的最后一行展示token使用信息
            stream_options={"include_usage": True}
        )
        result=""
        first = True
        for chunk in completion:
            if len(chunk.choices)>0:
                #print(chunk.choices[0].delta.content)
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos=0
                #msglist = re.split('[,.!;:，。！?]',msg)
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result)>10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result,datainfo)
                            result=""
                result = result+msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result,datainfo)
        
    except Exception as e:
        logger.exception('llm exceptiopn:')
        return   