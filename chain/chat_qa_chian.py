from operator import itemgetter

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document, PromptTemplate
from langchain_core.runnables import RunnableParallel

from embedding.dashscope_embedding import dashscope_embeddings
from llm.get_llm import get_llm


class RGAChain:
    """
    RGAChain用于创建一个带有对话历史纪录的检索增强生存（RGA）问答链
    """

    def __init__(self, docs: list, model_name: str = 'qwen-max', temperature: float = 0.1):
        """

        :param docs: 用于 bm25 检索的文字材料list
        :param model_name: llm 模型选择
        :param temperature: llm 模型 temperature 参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.chat_history = []
        self.docs = docs

    def clear_history(self):
        # 清空历史记录
        return self.chat_history.clear()

    def get_history(self):
        # 获取历史纪录
        return self.chat_history

    def answer(self, question: str):
        """
        answer方法用于调用 RGA 问答链，并返回问答结果
        :param question: 问题字符串
        :return: 问答结果
        """

        # 创建 embedding
        embeddings = dashscope_embeddings()

        # 构建向量检索器
        persist_directory = '../data_store/chroma_db'
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vector_retriever = vectordb.as_retriever()

        if self.docs == [] or len(self.docs) < 1:
            # 若传入的关键字检索内容不为空，则加入 bm25关键字检索，否则仅使用向量检索器
            retriever = vector_retriever
            print("向量检索")
        else:
            # 构建基于BM25的关键字检索器
            bm25_retriever = BM25Retriever.from_documents(self.docs)
            # 构建混合检索器
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5],
            )
            print("混合检索")

        # 构建模板

        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        仅输出standalone question的部分，其他内容请勿出现"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        template = """
                    根据以下内容,给出一个简明扼要的答复:

                    相关上下文信息:
                    {context}

                    对话历史:
                    {chat_history}

                    问题:
                    {question}

                    在生成答复时,请综合考虑上下文信息和对话历史,确保答复连贯、准确和相关。

                    如果问题是关于解决某个具体问题或完成任务,请在可能的情况下给出解决方案的步骤,并使用适当的代码块展示示例代码。

                    如果上下文信息不足以完整回答问题,请给出一个合理的推测,并明确指出缺乏确凿信息。

                    无论回答问题还是给出解决方案,都应该保持简洁明了,避免冗长或无谓的叙述。
                    """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

        def _combine_documents(
                docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        _inputs = CONDENSE_QUESTION_PROMPT | get_llm(model_name=self.model_name, temperature=0) | StrOutputParser()

        _context = {
            "context": itemgetter("standalone_question") | retriever | _combine_documents,
            "question": lambda x: x["standalone_question"],
            "chat_history": itemgetter("chat_history"),
        }

        conversational_qa_chain = (
                RunnableParallel({"standalone_question": _inputs,
                                  "chat_history": itemgetter("chat_history")})
                | _context
                | ANSWER_PROMPT
                | get_llm(model_name=self.model_name, temperature=self.temperature)
        )

        llm_input = {
            "question": question,
            "chat_history": str(self.chat_history)
        }
        answer = conversational_qa_chain.invoke(llm_input)

        self.chat_history.append({"human": question, "ai": answer})
        return answer
