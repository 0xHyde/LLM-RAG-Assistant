import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

_ = load_dotenv(find_dotenv())
dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]


def dashscope_embeddings(api_key:str=dashscope_api_key, model: str = "text-embedding-v2"):
    """
    获取DashScope embedding模型

    :param api_key: DashScope API key
    :param model: 模型名称，默认为 text-embedding-v2
    :return: embedding 模型
    """
    embeddings = DashScopeEmbeddings(
        model=model, dashscope_api_key=api_key
    )
    return embeddings