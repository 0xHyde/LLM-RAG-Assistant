import json
import os

from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from embedding.dashscope_embedding import dashscope_embeddings


def load_document_from_file(loaders: list, file_path: str):
    """
    从一个 pdf、txt 或 md 文档中加载内容。
    :param loaders: 已经加载的文件
    :param file_path: 目标文件地址
    """
    loader = None
    filename, extension = os.path.splitext(file_path)
    print("current file:", filename + extension)
    if extension == '.pdf':
        loader = PyMuPDFLoader(file_path).load()
    elif extension == '.md':
        loader = UnstructuredMarkdownLoader(file_path).load()
    elif extension == '.txt' or extension == 'yaml':
        loader = UnstructuredFileLoader(file_path).load()
    if loader is not None:
        loaders += loader


def load_document_from_dir_without_repetition(dir_path: str, saved_file_name_list: list, loaders: list = []):
    """
    从文件夹无重复的加载文件：检索已构建数据库的文件列表，匹配不重复的文件名，并加载到 loaders 返回
    :param dir_path: 文件夹路径
    :param saved_file_name_list: 已构建数据库的文件名列表
    :param loaders: 已存在的 loaders，文件内容会加载进其中
    :return: loaders
    """
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file not in saved_file_name_list:
                # os.path.join()用于组合目录和文件名得到完整路径
                file_path = os.path.join(root, file)
                load_document_from_file(loaders, file_path)
                add_to_saved_files(path='../data_store/saved_files.json', saved_file_name_list=saved_file_name_list,
                                   filename=file)
    return loaders


def load_saved_files(path: str = '../data_store/saved_files.json'):
    # 加载已构建数据库的文件名列表
    with open(path, 'r') as f:
        # 使用json.load()函数读取并解析文件内容
        data = json.load(f)
        return data


def add_to_saved_files(saved_file_name_list: list, filename: str, path: str = '../data_store/saved_files.json'):
    # 将 filename 添加进已构建数据库的文件名列表，并保存至 json
    saved_file_name_list.append(filename)
    with open(path, 'w') as f:
        json.dump(saved_file_name_list, f)


def build_vector_database(split_docs: list, embedding, persist_directory: str):
    # 加载数据库
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_directory)
    # 向量持久化
    vectordb.persist()


def get_docs_from_txt(txt_path: str = '../data_store/bm25_store.txt'):
    # 读取txt并创建doc数据
    docs = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            docs.append(Document(line))
    return docs


def build_custom_database(dir_path: str = '../data_store/documents'):
    print('Building custom database...')
    saved_file_name_list = load_saved_files()

    # 加载并切分文档
    loaders = load_document_from_dir_without_repetition(dir_path=dir_path,
                                                        saved_file_name_list=saved_file_name_list)
    print("Document loading completed, length:", len(loaders))

    print("In the document slicing")
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(loaders)
    print("Document slicing is completed")

    # 构建文本存储
    print("Build in the text storage")
    with open('../data_store/bm25_store.txt', 'a') as f:
        for line in split_docs:
            f.write(str(line))
            f.write("\n")

    # 构建向量存储
    print("Build in the vector storage")
    # 定义 Embeddings
    embedding = dashscope_embeddings()
    # 加载数据库
    for i in range(0, len(split_docs), 100):
        build_vector_database(split_docs[i:i + 100], embedding, persist_directory='../data_store/chroma_db')
        print(f"{i + 100}/{len(split_docs)}")

    print("Database construction completed")