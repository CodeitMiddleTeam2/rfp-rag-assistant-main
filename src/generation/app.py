import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

#==============================================
# 프로그램명: app.py
# 폴더위치: src/generation/app.py
# 프로그램 설명: RAG 기반 RFP 분석 플랫폼 (DB + Rerank + Local LLM)
# 작성이력: 2025.12.19 한상준 최초 작성
#          12.21 수정 : 한상준 대분류 종합모드 추가
#          12.23 수정 : 한상준 DB 연동 코드 추가
#          12.24 수정 : 한상준 rerank 추가
#          12.29 수정 : src/rag/db.py rerank_model.py embedding_model.py 병합
#===============================================

# [1. 환경 변수 및 경로 설정]
load_dotenv()

current_file = os.path.abspath(__file__)
generation_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(generation_dir)
root_dir = os.path.dirname(src_dir)
model_path = os.path.join(root_dir, "unsloth.Q4_K_M.gguf")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# [2. 모듈 임포트]
try:
    from src.prompts.RAGPromptBuilder import RAGPromptBuilder
    from src.generation.model_manager import ModelManager
    from src.rag.embed.embedding_model import EmbeddingModel
    from src.rag.rerank.rerank_model import RerankModel
except ImportError as e:
    st.error(f"❌ 모듈 임포트 실패: {e}")
    st.stop()

# [3. 데이터 로드 함수]
def load_hierarchical_data():
    csv_path = os.path.join(root_dir, 'final_classification_hierarchy.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("🚨 계층 데이터 파일이 없습니다.")
        return None

def main():
    st.set_page_config(page_title="RFP Intelligence Platform", layout="wide", page_icon="🏢")

    # 데이터 로드 (사이드바 필터용)
    df = load_hierarchical_data()
    if df is None: return

    # ✅ 매니저 인스턴스 초기화
    # ModelManager는 내부 캐싱되므로 매번 호출해도 안전함
    model_manager = ModelManager(local_model_path=model_path)

    # ✅ Advanced RAG 모듈 초기화
    try:
        embedding_model = EmbeddingModel("text-embedding-3-small")
        rerank_model = RerankModel("dragonkue/bge-reranker-v2-m3-ko") # L4 GPU 자동 사용됨
    except Exception as e:
        st.error(f"❌ RAG 모델 초기화 실패: {e}")
        st.stop()

    # 프롬프트 빌더 초기화
    try:
        prompt_dir = os.path.join(root_dir, 'src', 'prompts')
        builder = RAGPromptBuilder(prompt_dir)
    except:
        st.warning("⚠️ 프롬프트 빌더 초기화 실패. 기본 모드로 동작합니다.")
        builder = None

    st.title("🏢 B2G 입찰 분석 플랫폼: 계층형 탐색 모드")
    st.markdown("카테고리별로 사업을 탐색하고, **DB 기반의 정밀 RAG 분석**을 수행합니다.")

    # ---------------------------------------------------------
    # [Sidebar] 설정 및 필터
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 모델 설정")
        model_source = st.radio("사용 모델", ("OpenAI API (GPT-5-nano)", "Local Model (Qwen-3-8B)"), index=0)
        
        openai_client = None
        local_llm = None
        source_key = "openai"

        if "OpenAI" in model_source:
            source_key = "openai"
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

        display_title = ""
        # ⚠️ 필터링 로직: DB 검색을 위해 '선택된 사업명'을 추적해야 함
        target_project_name_for_db = "%" # 기본값: 전체 검색

        if selected_d1 == "🔍 전체 데이터 (All RFPs)":
            display_title = "전체 RFP 데이터 종합 분석"
            selected_d2 = None
            selected_project = None
        else:
            # --- Depth 2: 중분류 ---
            d2_options = ["📂 해당 대분류 전체 종합"] + sorted(df[df['Depth_1'] == selected_d1]['Depth_2'].unique().tolist())
            selected_d2 = st.selectbox("2단계: 중분류", d2_options)

            if selected_d2 == "📂 해당 대분류 전체 종합":
                display_title = f"[{selected_d1}] 카테고리 전체 분석"
            else:
                # --- Depth 3: 프로젝트 ---
                projects_in_cat = df[(df['Depth_1'] == selected_d1) & (df['Depth_2'] == selected_d2)]
                proj_options = ["🎁 해당 중분류 전체 종합"] + sorted(projects_in_cat['사업명'].tolist())
                selected_project = st.selectbox("3단계: 상세 사업", proj_options)

                if selected_project == "🎁 해당 중분류 전체 종합":
                    display_title = f"[{selected_d2}] 하위 사업 전체 분석"
                else:
                    display_title = selected_project
                    target_project_name_for_db = selected_project # ✅ 특정 사업 선택 시 필터 적용

    # ---------------------------------------------------------
    # [Main] UI 레이아웃
    # ---------------------------------------------------------
    
    col_info, col_chat = st.columns([1, 1.5])

    with col_info:
        st.subheader(f"📊 {display_title}")

        st.info("💡 질문을 입력하면 DB에서 가장 관련성 높은 문서를 찾아 답변합니다.")
        st.markdown(f"**현재 검색 필터:** `{target_project_name_for_db if target_project_name_for_db != '%' else "전체 범위"}`")

    with col_chat:
        st.subheader("💬 AI 컨설턴트 질의응답")

        chat_container = st.container(height=600)
        
        with chat_container:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 채팅 로그 출력
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # 질문 입력
        if query := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": query})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)

                # 답변 생성
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("⏳ DB 검색 진행 중...")

                    try:
                        # ✅ [수정 1] DB 검색 호출 (Threshold 설정)
                        # 필터 기능이 없는 함수이므로, 일단 넉넉하게(30~50개) 가져옵니다.
                        initial_results = embedding_model.search(
                            query=query, 
                            result_count=40, # 필터링을 위해 넉넉히 조회
                            threshold=0.3    # 유사도 0.3 이상만
                        )

                        for doc in initial_results:
                            if 'text' in doc:
                                doc['content'] = doc['text']
                        
                        # ✅ [수정 2] 파이썬 레벨에서 필터링 (DB 함수가 지원 안 하므로 수동 처리)
                        filtered_results = []

                        if target_project_name_for_db == "%":
                            filtered_results = initial_results
                        else:
                            for doc in initial_results:
                                # 1. DB 테이블의 컬럼('project_name') 직접 확인 (가장 정확)
                                p_name = doc.get('project_name')
                                
                                # 2. 혹시 몰라 메타데이터 안쪽도 확인 (이전 호환성)
                                if not p_name:
                                    p_name = doc.get('metadata', {}).get('project_name')

                                # 3. 사이드바에서 선택한 사업명과 비교
                                # (DB에는 띄어쓰기가 다를 수 있으므로 공백 제거 후 비교하는 게 안전할 수 있음)
                                if p_name and p_name == target_project_name_for_db:
                                    filtered_results.append(doc)

                        # (디버깅용) 필터링 전후 개수 확인
                        st.write(f"검색된 {len(initial_results)}개 중 '{target_project_name_for_db}' 관련 문서 {len(filtered_results)}개 필터링 됨")

                        retrieval_results = filtered_results

                        if not retrieval_results:
                            combined_context = "조건에 맞는 문서를 찾을 수 없습니다."
                        else:
                            # 2. Reranking (상위 3개)
                            reranked_result_obj = rerank_model.rerank(
                                query, 
                                retrieval_results, 
                                top_k=3
                            )
                            combined_context = reranked_result_obj.content

                            # [디버깅] Rerank 점수 및 메타데이터 확인
                            with st.expander("🔍 Rerank 결과 상세 보기"):
                                st.text(combined_context)

                        # ✅ 프롬프트 조립
                        if builder:
                            final_messages = builder.build_messages(
                                category=selected_d1 if selected_d1 else "General",
                                title=display_title,
                                context=combined_context,
                                history=st.session_state.messages[:-1],
                                query=query
                            )
                        else:
                            # Fallback
                            final_messages = [
                                {"role": "system", "content": "당신은 입찰 전문가입니다."},
                                {"role": "user", "content": f"참고문서:\n{combined_context}\n\n질문: {query}"}
                            ]

                        # ✅ 답변 생성
                        message_placeholder.markdown("⏳ 답변 생성 중...")
                        response_text = model_manager.generate_response(
                            messages=final_messages,
                            source=source_key,
                            local_llm=local_llm,
                            openai_client=openai_client
                        )
                        
                        message_placeholder.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                    except Exception as e:
                        if "CUDA out of memory" in str(e):
                            st.error("🚨 GPU 메모리 부족! 잠시 후 다시 시도해주세요.")
                            model_manager.clear_gpu_memory() # 👈 메모리 청소 실행
                            st.stop()
                        message_placeholder.error(f"❌ 에러 발생: {str(e)}")

if __name__ == "__main__":
    main()