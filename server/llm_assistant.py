import sys

sys.path.append("../")

import streamlit as st
from data_base.database_build import get_docs_from_txt
from chain.chat_qa_chian import RGAChain

if "chain" not in st.session_state:
    # 加载大模型链
    docs = get_docs_from_txt()
    chain = RGAChain(docs=docs)
    st.session_state.chain = chain

chain = st.session_state.chain

st.set_page_config(
    page_title="LLM助手",
    page_icon="👋",
)

# 侧边栏
st.sidebar.subheader('LLM模型配置')

# 选择模型
model_server = st.sidebar.selectbox(
    'LLM 服务商',
    ['DashScope (阿里云)'])

if model_server == 'DashScope (阿里云)':
    model_name = st.sidebar.selectbox(
        '模型名称',
        ['qwen-max', 'qwen-max-longcontext', 'qwen-1.8b-chat'])

# 将模型选择同步至 chain
chain.model_name = model_name

# 设置 temperature
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.1
)
# 同步 temperature 至 chain
chain.temperature = temperature

st.sidebar.divider()
st.sidebar.subheader('聊天设置')

# 清除历史纪录按钮
click_clear_history = st.sidebar.button('清除历史记录', use_container_width=True)
if click_clear_history:
    chain.clear_history()

# 主页面

# 加载历史信息
chat_history = chain.chat_history
for message in chat_history:
    with st.chat_message("human"):
        st.write(message["human"])
    with st.chat_message("ai"):
        st.write(message["ai"])

prompt = st.chat_input(placeholder="Your message")
if prompt:
    with st.chat_message("human"):
        st.write(prompt)
    response = chain.answer(prompt)
    with st.chat_message("ai"):
        st.write(response)