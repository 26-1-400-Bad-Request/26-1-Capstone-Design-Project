import os
import sys
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))        
from src.rag import create_rag_chain

st.set_page_config(page_title="건축 인허가 AI (Ollama)", page_icon="🏢", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 사이드바
with st.sidebar:
    st.header("🦙 Ollama 설정")
    model_option = st.text_input("사용할 Ollama 모델명", value="exaone3.5:7.8b")        # 사용중인 모델명 작성 
    search_k = st.slider("참조 문서 개수", 1, 10, 4)
    st.info("Ollama가 PC에서 실행 중이어야 합니다.")
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

st.title("🏗️ 건축 법령 RAG 보좌관 (로컬 실행)")

# 대화 로그 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력
if user_query := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_area = st.empty()
        try:
            with st.spinner("로컬 AI가 법령을 분석 중입니다..."):
                chain, retriever = create_rag_chain(model_name=model_option, search_k=search_k)
                
                # get_relevant_documents 대신 invoke 사용
                relevant_chunks = retriever.invoke(user_query) 
                
                ai_answer = chain.invoke(user_query)
                
                response_area.markdown(ai_answer)
                
                with st.expander("🔍 근거 문서 확인"):
                    for idx, doc in enumerate(relevant_chunks):
                        st.write(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source',''))}**")
                        st.caption(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}\nOllama가 켜져 있는지 확인해 주세요!")