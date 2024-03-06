from __future__ import annotations

from typing import List

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel
from xinference.client import Client


class CustomEmbeddings(BaseModel, Embeddings):
    # 自定义embedding类，本处使用Xinference运行bge-m3模型

    def _embed(self, texts: str) -> List[float]:
        # 生成输入文本的 embedding。
        # Args:
        #     texts (str): 要生成 embedding 的文本。
        # Return:
        #     embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表。
        client = Client("http://localhost:9997")
        model = client.get_model("bge-m3-local")
        # model = client.get_model("bge-base-zh-v1.5")
        response = model.create_embedding(texts)
        return response['data'][0]['embedding']

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 生成输入文本列表的 embedding。
        # Args:
        #     texts (List[str]): 要生成 embedding 的文本列表.
        # Returns:
        #     List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding。

        Args:
            text (str): 要生成 embedding 的文本。

        Return:
            List [float]: 输入文本的 embedding，一个浮点数值列表。
        """
        resp = self.embed_documents([text])
        return resp[0]
