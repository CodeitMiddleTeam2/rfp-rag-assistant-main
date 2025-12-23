import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# [1. 환경 변수 로드]
load_dotenv()

# [2. 경로 설정 및 모듈 임포트]
current_file = os.path.abspath(__file__)
generation_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(generation_dir)
root_dir = os.path.dirname(src_dir)
model_path = os.path.join(root_dir, "unsloth.Q4_K_M.gguf")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 💡 안전한 임포트 관리
try:
    from src.prompts.RAGPromptBuilder import RAGPromptBuilder
    from src.generation.model_manager import ModelManager
    from src.generation.supabase_manager import SupabaseManager
except ImportError as e:
    st.error(f"❌ 모듈 임포트 실패: {e}")
    st.stop()

# [2. 데이터 로드 및 초기화]
def load_hierarchical_data():
    csv_path = os.path.join(root_dir, 'final_classification_hierarchy.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("🚨 계층 데이터 파일이 없습니다.")
        return None

def main():
    st.set_page_config(page_title="RFP Intelligence Platform", layout="wide", page_icon="🏢")
    df = load_hierarchical_data()
    if df is None: return

    # ✅ [객체 생성] ModelManager 인스턴스화
    # 인스턴스는 매 실행마다 새로 생성되지만, 내부에서 호출하는
    # _load_llama_cpp_model 함수가 캐싱되어 있어 모델은 1번만 로드됩니다.
    model_manager = ModelManager(local_model_path=model_path)
    db_manager = SupabaseManager()

    try:
        prompt_dir = os.path.join(root_dir, 'src', 'prompts')
        builder = RAGPromptBuilder(prompt_dir)
    except:
        st.warning("⚠️ RAGPromptBuilder를 찾을 수 없어 중지합니다.")
        builder = None

    st.title("🏢 B2G 입찰 분석 플랫폼: 계층형 탐색 모드")
    st.markdown("카테고리별로 사업을 탐색하고, **여러 사업을 동시에 비교/분석**할 수 있습니다.")

    # ---------------------------------------------------------
    # [Sidebar] 설정 및 필터
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 모델 설정")
        model_source = st.radio("사용 모델", ("OpenAI API (GPT-5-mini)", "Local Model (Qwen-3-8B)"), index=0)
        
        openai_client = None
        local_llm = None

        if "OpenAI" in model_source:
            source_key = "openai"
            # ✅ API Key 검증을 여기서 수행 (로컬 유저는 통과 가능)
            if not os.getenv("OPENAI_API_KEY"):
                st.error("🚨 .env 파일에 OPENAI_API_KEY가 없습니다.")
            else:
                openai_client = model_manager.get_openai_client()
                st.success("🟢 API Ready")
        else:
            source_key = "local"
            with st.spinner("🚀 로컬 모델 로딩 중..."):
                # ✅ 매니저를 통해 모델 로드 (내부적으로 캐싱됨)
                local_llm = model_manager.load_local_model()
            if local_llm: 
                st.success("🟢 Local Model Ready")
            else:
                st.error(f"❌ 모델 로드 실패. 경로 확인: {model_path}")

        st.divider()

        st.header("📂 탐색 필터")

        # --- Depth 1: 대분류 ---
        d1_options = ["🔍 전체 데이터 (All RFPs)"] + sorted(df['Depth_1'].unique().tolist())
        selected_d1 = st.selectbox("1단계: 대분류", d1_options)

        target_rows = pd.DataFrame() # 분석 대상 데이터
        display_title = ""

        if selected_d1 == "🔍 전체 데이터 (All RFPs)":
            # 전체 모드: 하위 옵션 비활성화
            target_rows = df
            display_title = "전체 RFP 데이터 종합 분석"
            st.info("⚠️ 전체 문서는 양이 많아 분석이 느릴 수 있습니다.")
            selected_d2 = None
            selected_project = None
        else:
            # --- Depth 2: 중분류 ---
            d2_options = ["📂 해당 대분류 전체 종합"] + sorted(df[df['Depth_1'] == selected_d1]['Depth_2'].unique().tolist())
            selected_d2 = st.selectbox("2단계: 중분류", d2_options)

            if selected_d2 == "📂 해당 대분류 전체 종합":
                target_rows = df[df['Depth_1'] == selected_d1]
                display_title = f"[{selected_d1}] 카테고리 전체 분석"
                selected_project = None
            else:
                # --- Depth 3: 프로젝트 ---
                projects_in_cat = df[(df['Depth_1'] == selected_d1) & (df['Depth_2'] == selected_d2)]
                proj_options = ["🎁 해당 중분류 전체 종합"] + sorted(projects_in_cat['사업명'].tolist())
                selected_project = st.selectbox("3단계: 상세 사업", proj_options)

                if selected_project == "🎁 해당 중분류 전체 종합":
                    target_rows = projects_in_cat
                    display_title = f"[{selected_d2}] 하위 사업 전체 분석"
                else:
                    target_rows = df[df['사업명'] == selected_project]
                    display_title = selected_project

    # ---------------------------------------------------------
    # [Main] 컨텍스트 조립 및 UI
    # ---------------------------------------------------------
    
    filter_metadata = {}
    if selected_d1 != "🔍 전체 데이터 (All RFPs)":
        filter_metadata['depth_1'] = selected_d1
    if selected_d2 and selected_d2 != "📂 해당 대분류 전체 종합":
        filter_metadata['depth_2'] = selected_d2
    if selected_project and selected_project != "🎁 해당 중분류 전체 종합":
        filter_metadata['project_name'] = selected_project

    # [UI 레이아웃]
    col_info, col_chat = st.columns([1, 1.5])

    with col_info:
        st.subheader(f"📊 {display_title}")
        st.caption(f"참조 문서: {len(target_rows)}건")

    with col_chat:
        st.subheader("💬 AI 컨설턴트 질의응답")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 채팅 로그 출력
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 질문 입력
        if query := st.chat_input("질문을 입력하세요 (예: 이 카테고리 사업들의 공통적인 자격 요건은 뭐야?)"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # 답변 생성
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ DB에서 관련 문서를 찾는 중...")

                try:
                    # ✅ 1. Supabase 벡터 검색 (RAG 핵심)
                    # 필터링 조건에 맞는 문서 중, 질문과 관련된 Top 5 청크만 가져옴
                    retrieved_docs = db_manager.similarity_search(
                        query=query, 
                        filters=filter_metadata, # 이 필터는 RPC 함수 구현에 따라 적용 방식이 다름
                        top_k=5
                    )
                    
                    if not retrieved_docs:
                        combined_context = "관련된 정보를 데이터베이스에서 찾을 수 없습니다."
                    else:
                        combined_context = db_manager.format_docs(retrieved_docs)
                        # 디버깅: 검색된 청크 보여주기 (선택사항)
                        with st.expander("🔍 검색된 RAG 컨텍스트 확인"):
                            st.write(combined_context)

                    # ✅ 2. 프롬프트 조립
                    if builder:
                        final_messages = builder.build_messages(
                            category=selected_d1 if selected_d1 else "General",
                            title=display_title,
                            context=combined_context, # 여기가 이제 전체 텍스트가 아니라 검색된 텍스트임
                            history=st.session_state.messages[:-1],
                            query=query
                        )
                    else:
                        # Fallback
                        final_messages = [
                            {"role": "system", "content": "당신은 입찰 전문가입니다."},
                            {"role": "user", "content": f"참고문서:\n{combined_context}\n\n질문: {query}"}
                        ]

                    # ✅ 3. 답변 생성 (기존 로직 동일)
                    response_text = model_manager.generate_response(
                        messages=final_messages,
                        source=source_key,
                        local_llm=local_llm,
                        openai_client=openai_client
                    )
                    
                    message_placeholder.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    message_placeholder.error(f"❌ 에러 발생: {str(e)}")

if __name__ == "__main__":
    main()