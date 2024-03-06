import random
from http import HTTPStatus
from typing import Any, List, Mapping, Optional

import dashscope
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


def get_llm_qwen(model_name: str = "qwen-max", temperature: float = "0.1"):
    llm = llm_qwen()
    if llm.model_name is not model_name:
        llm.model_name = model_name
    if llm.temperature is not temperature:
        llm.temperature = temperature
    return llm


class llm_qwen(LLM):
    # 默认选用 qwen-max 模型
    model_name = 'qwen-max'
    # 温度系数
    temperature = 0.1

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:

        # 构造用户提示
        messages = [{'role': 'user', 'content': prompt}]

        # 通过SDK调用大模型
        response = dashscope.Generation.call(
            # 选择模型
            model=self.model_name,
            # 传递用户提示
            messages=messages,
            # 温度系数
            temperature=self.temperature,
            # set the random seed, optional, default to 1234 if not set
            seed=random.randint(1, 10000),
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}
