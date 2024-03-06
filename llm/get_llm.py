import os

import dashscope
from dotenv import find_dotenv, load_dotenv

from llm.llm_qwen import get_llm_qwen


def get_llm(model_name: str,
            temperature=0.1,
            api_key=None,
            secret_key=None,
            access_token=None,
            appid=None,
            api_secret=None):
    if model_name in ['qwen-max', 'qwen-max-longcontext', 'qwen-1.8b-chat']:
        # 配置API key
        _ = load_dotenv(find_dotenv())
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        return get_llm_qwen(model_name, temperature)
    else:
        return "不正确的模型"
