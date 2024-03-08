import os
import sys

import streamlit as st

sys.path.append("../")

from data_base.database_build import load_saved_files, build_custom_database, get_docs_from_txt

saved_files = load_saved_files()

# 侧边栏
st.sidebar.subheader('数据库文档：')
for saved_file in saved_files:
    st.sidebar.write(saved_file)

# 主页面
st.header("")
# 文件上传组件
uploaded_files = st.file_uploader("请上传 pdf、markdown 或 txt 文档", accept_multiple_files=True)
if uploaded_files != [] or len(uploaded_files) > 0:
    click = st.button("构建数据库", use_container_width=True)
    if click:
        with st.spinner('正在构建数据库中，请等待'):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                save_path = os.path.join("../data_store/documents", file_name)
                with open(save_path, 'wb') as f:
                    # 读取上传文件的内容并写入到新文件
                    f.write(uploaded_file.getvalue())
            build_custom_database()
            st.session_state.chain.docs = get_docs_from_txt()
        st.toast('数据库构建完成！', icon='🎉')