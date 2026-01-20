import streamlit as st
import json
import os
from dotenv import load_dotenv

# 최신 라이브러리 임포트 (지시사항 반영)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 페이지 설정
st.set_page_config(page_title="FAQ 지능형 챗봇", page_icon="🤖", layout="wide")

# --- 2. 데이터 준비 및 벡터 DB (캐싱 처리) ---
@st.cache_resource
def get_vector_db():
    file_path = 'data/faq_chatbot_data.json'
    
    if not os.path.exists(file_path):
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다.")
        st.stop()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    documents = []
    for item in faq_data:
        # 지시사항 준수: 질문과 답변을 결합하여 임베딩
        combined_content = f"질문: {item['question']}\n답변: {item['answer']}"
        doc = Document(
            page_content=combined_content, 
            metadata={"answer": item['answer'], "question": item['question']}
        )
        documents.append(doc)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db, faq_data

# --- 3. 메인 로직 ---
def main():
    st.title("🏢 고객지원 지능형 FAQ 센터")
    st.markdown("자주 묻는 질문을 선택하거나, 궁금한 점을 직접 물어보세요.")
    
    if not api_key:
        st.error("`.env` 파일에 API 키를 설정해 주세요.")
        return

    # DB 및 데이터 로드
    vector_db, raw_faq = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # UI 레이아웃 분리
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 FAQ 리스트에서 선택")
        faq_questions = [f["question"] for f in raw_faq]
        selected_faq = st.selectbox("질문을 선택하세요:", ["선택 안 함"] + faq_questions)

    with col2:
        st.subheader("✍️ 직접 질문 입력")
        user_query = st.text_input("질문을 입력하세요:", placeholder="예: 배송은 보통 얼마나 걸리나요?")

    # 최종 질문 결정
    final_query = ""
    if user_query: # 직접 입력이 있으면 우선순위
        final_query = user_query
    elif selected_faq != "선택 안 함":
        final_query = selected_faq

    if final_query:
        st.write(f"**🔍 질문 내용:** {final_query}")
        
        # --- 4. RAG 체인 구성 (LCEL 최신 문법) ---
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        prompt = ChatPromptTemplate.from_template("""
        당신은 친절한 고객센터 상담원입니다. 제공된 [FAQ 정보]를 바탕으로 정중하게 답변해 주세요.
        정보가 없다면 "죄송하지만 해당 내용은 상담원 연결이 필요합니다(1588-0000)."라고 안내하세요.

        [FAQ 정보]
        {context}

        질문: {input}
        답변:
        """)

        # 문서를 텍스트로 합쳐주는 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL 파이프라인
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 답변 생성
        with st.spinner('답변을 생성 중입니다...'):
            try:
                # 1. 챗봇 답변 생성
                response = rag_chain.invoke(final_query)
                
                st.success("### 📢 상담원 답변")
                st.write(response)

                # 2. 검색 근거 확인 (에러 해결: get_relevant_documents -> invoke)
                with st.expander("🔍 검색된 관련 FAQ 데이터 (근거)"):
                    # 최신 버전에서는 invoke를 사용합니다.
                    relevant_docs = retriever.invoke(final_query)
                    for i, doc in enumerate(relevant_docs):
                        st.info(f"**관련 정보 {i+1}**\n\n{doc.page_content}")
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()