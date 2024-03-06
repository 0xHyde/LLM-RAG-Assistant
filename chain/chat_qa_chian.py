from operator import itemgetter

from langchain.chains import LLMChain
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever, LineListOutputParser
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document, PromptTemplate
from langchain_core.runnables import RunnableParallel

from embedding.custom_embedding import CustomEmbeddings
from llm.get_llm import get_llm


class RGA_Chain():

    def __init__(self, model_name: str = 'qwen-max', temperature: float = 0.1, chat_history: list = [],
                 docs: list = []):
        self.model_name = model_name
        self.temperature = temperature
        self.chat_history = []
        self.docs = []

    def clear_history(self):
        # 清空历史记录
        return self.chat_history.clear()

    def get_history_str(self):
        return get_buffer_string(self.chat_history)

    def set_history_ai(self, response):
        # 将ai相应写入history
        self.chat_history.append(AIMessage(content=response))

    def set_history_human(self, prompt):
        # 将ai相应写入history
        self.chat_history.append(HumanMessage(content=prompt))

    def get_docs_from_txt(self, txt_path):
        # 读取txt并创建doc数据
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.docs.append(Document(line))
        except FileNotFoundError:
            print("Error: No file found or failed to read the file")
        else:
            print("loaded successfully")

    def answer(self, question: str):

        # 构建数据检索
        embedding = CustomEmbeddings()

        persist_directory = '../chroma_db'
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

        # 构建基于BM25的关键字检索器
        bm25_retriever = BM25Retriever.from_documents(self.docs)

        # 构建基于向量的检索器
        vector_retriever = vectordb.as_retriever()

        # 构建混合检索器
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

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
            "chat_history": self.get_history_str()
        }
        answer = conversational_qa_chain.invoke(llm_input)

        self.set_history_human(question)
        self.set_history_ai(answer)

        return answer
