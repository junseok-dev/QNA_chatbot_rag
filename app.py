import streamlit as st
import json
import os
from dotenv import load_dotenv

# 최신 라이브러리 임포트
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 페이지 설정
st.set_page_config(page_title="FAQ RAG 챗봇", page_icon="🔑", layout="wide")

# 사이드바에서 API 키 입력 받기
st.sidebar.title("🔐 설정")
user_api_key = st.sidebar.text_input(
    "OpenAI API Key를 입력하세요", 
    type="password", 
    placeholder="sk-..."
)
st.sidebar.info("입력하신 API 키는 메모리에만 유지되며 저장되지 않습니다.")

# --- 1. 데이터 로드 및 벡터 DB 생성 (API 키 필요) ---
@st.cache_resource
def get_vector_db(api_key):
    # API 키가 없으면 실행하지 않음
    if not api_key:
        return None, None
        
    file_path = 'faq_chatbot_data.json'
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다.")
        st.stop()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    documents = []
    for item in faq_data:
        # 지시사항: 질문+답변 결합 임베딩
        combined_content = f"질문: {item['question']}\n답변: {item['answer']}"
        doc = Document(
            page_content=combined_content, 
            metadata={"answer": item['answer'], "question": item['question']}
        )
        documents.append(doc)
    
    # 전달받은 API 키로 임베딩 모델 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db, faq_data

# --- 2. 메인 화면 로직 ---
def main():
    st.title("🏢 지능형 FAQ 센터 (RAG)")
    
    # API 키가 입력되지 않았을 때의 안내
    if not user_api_key:
        st.warning("👈 왼쪽 사이드바에 OpenAI API Key를 입력해 주세요.")
        return

    # 데이터 및 리트리버 준비
    vector_db, raw_faq = get_vector_db(user_api_key)
    if not vector_db:
        return
        
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # UI 레이아웃 분리
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 FAQ 리스트에서 선택")
        faq_questions = [f["question"] for f in raw_faq]
        selected_faq = st.selectbox("리스트 중 하나를 선택하세요:", ["선택 안 함"] + faq_questions)

    with col2:
        st.subheader("✍️ 직접 질문 입력 (주관적 질문)")
        user_query = st.text_input("궁금한 내용을 자유롭게 입력하세요:", placeholder="예: 반품 기간이 어떻게 되나요?")

    # 최종 질문 결정 (직접 입력 우선)
    final_query = ""
    if user_query:
        final_query = user_query
    elif selected_faq != "선택 안 함":
        final_query = selected_faq

    # --- 3. RAG 실행 파이프라인 (LCEL) ---
    if final_query:
        st.info(f"**현재 질문:** {final_query}")
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=user_api_key)

        prompt = ChatPromptTemplate.from_template("""
        당신은 친절한 고객센터 상담원입니다. 제공된 [FAQ 정보]를 바탕으로 정중하게 답변해 주세요.
        정보가 없다면 "죄송하지만 해당 내용은 상담원 연결이 필요합니다."라고 안내하세요.

        [FAQ 정보]
        {context}

        질문: {input}
        답변:
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 결과 출력
        with st.spinner('답변을 생성 중입니다...'):
            try:
                # 1. 메인 답변
                response = rag_chain.invoke(final_query)
                st.success("### 📢 상담원 답변")
                st.write(response)

                # 2. 검색 근거 확인 (에러 해결: invoke 메서드 사용)
                with st.expander("🔍 관련 FAQ 검색 근거 데이터"):
                    relevant_docs = retriever.invoke(final_query)
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**관련 정보 {i+1}**")
                        st.caption(doc.page_content)
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                st.info("API 키가 올바른지, 혹은 결제 한도가 초과되지 않았는지 확인해 주세요.")

if __name__ == "__main__":
    main()

