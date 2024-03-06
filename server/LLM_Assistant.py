import streamlit as st
import sys
sys.path.append("../")
from chain.chat_qa_chian import RGA_Chain

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model_name' not in st.session_state:
    st.session_state.model_name = ''

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.1

#加载大模型
llm = RGA_Chain(model_name=st.session_state.model_name,temperature=st.session_state.temperature)
llm.get_docs_from_txt('../document_store.txt')

st.set_page_config(
    page_title="LLM助手",
    page_icon="👋",
)

# 侧边栏
st.sidebar.subheader('LLM模型配置')

model_server = st.sidebar.selectbox(
    'LLM 服务商',
    ['DashScope (阿里云)'])

if model_server == 'DashScope (阿里云)':
    model_name = st.sidebar.selectbox(
        '模型名称',
        ['qwen-max', 'qwen-max-longcontext', 'qwen-1.8b-chat'])

st.session_state.model_name = model_name

temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.1
)
st.session_state.temperature = temperature



# 主页面

# 加载历史信息并同步至llm
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["context"])
    if message["role"] == 'ai':
        llm.set_history_ai(message["context"])
    elif message["role"] == 'human':
        llm.set_history_human(message["context"])



prompt = st.chat_input(placeholder="Your message")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "human", "context": prompt})
    response = llm.answer(prompt)
    with st.chat_message("ai"):
        st.write(response)
    st.session_state.chat_history.append({"role":"ai","context":response})