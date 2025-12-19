import streamlit as st
import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# [1. 환경 변수 로드]
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("🚨 API Key가 없습니다. .env 파일을 확인해주세요!")
    st.stop() # 키가 없으면 앱 실행 중단

client = OpenAI(api_key=API_KEY)

# [2. 경로 설정 및 모듈 임포트]
current_file = os.path.abspath(__file__)
generation_dir = os.path.dirname(current_file)
src_dir = os.path.dirname(generation_dir)
root_dir = os.path.dirname(src_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 💡 안전한 임포트 관리
try:
    from src.prompts.RAGPromptBuilder import RAGPromptBuilder
    from get_llm_response import get_llm_response_safe
    # print("✅ 모든 모듈 임포트 성공!") # 디버깅용
except ImportError as e:
    st.error(f"❌ 모듈 임포트 실패: {e}")
    st.stop()

# [2. 데이터 로드 및 초기화]
def load_hierarchical_data():
    csv_path = os.path.join(root_dir, 'final_classification_hierarchy.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("🚨 계층 데이터 파일이 없습니다. 먼저 데이터 보정 스크립트를 실행하세요.")
        return None

def main():
    st.set_page_config(page_title="RFP Intelligence Platform", layout="wide", page_icon="🏢")
    df = load_hierarchical_data()
    if df is None: return

    # 프롬프트 빌더 초기화
    prompt_dir = os.path.join(root_dir, 'src', 'prompts')
    builder = RAGPromptBuilder(prompt_dir)

    st.title("🏢 B2G 입찰 분석 플랫폼: 계층형 탐색 모드")
    st.markdown("카테고리별로 사업을 탐색하고, **여러 사업을 동시에 비교/분석**할 수 있습니다.")

    # [3. 사이드바: 계층형 필터링]
    with st.sidebar:
        st.header("📂 카테고리 필터")
        
        # Depth 1: 대분류
        d1_list = sorted(df['Depth_1'].unique())
        selected_d1 = st.selectbox("1단계: 대분류 선택", d1_list)
        
        # Depth 2: 중분류 (Depth 1에 종속)
        d2_list = sorted(df[df['Depth_1'] == selected_d1]['Depth_2'].unique())
        selected_d2 = st.selectbox("2단계: 중분류 선택", d2_list)
        
        # 프로젝트 선택 (전체 선택 옵션 추가로 '여러 문서 종합' 대응)
        projects_in_cat = df[(df['Depth_1'] == selected_d1) & (df['Depth_2'] == selected_d2)]
        project_list = ["🎁 해당 카테고리 전체 분석 (종합 모드)"] + sorted(projects_in_cat['사업명'].tolist())
        selected_project = st.selectbox("3단계: 상세 사업 선택", project_list)

        st.divider()
        st.caption(f"📍 현재 위치: {selected_d1} > {selected_d2}")

    # [4. 컨텍스트 조립 로직 (성능 평가 2번 핵심)]
    if "전체 분석" in selected_project:
        # 여러 문서를 합치는 경우
        target_rows = projects_in_cat
        is_multi = True
        # 각 문서의 앞부분 1500자씩 발췌하여 결합 (토큰 관리)
        combined_context = ""
        for _, row in target_rows.iterrows():
            combined_context += f"### 사업명: {row['사업명']}\n{row['텍스트'][:1500]}\n\n"
        display_title = f"{selected_d2} 카테고리 전체 요약 분석"
    else:
        # 단일 문서인 경우
        target_row = df[df['사업명'] == selected_project].iloc[0]
        is_multi = False
        combined_context = target_row['텍스트']
        display_title = selected_project

    # [5. 메인 레이아웃]
    col_info, col_chat = st.columns([1, 1.2])

    with col_info:
        st.subheader(f"📊 {display_title}")
        if is_multi:
            st.warning(f"💡 현재 {len(target_rows)}개의 사업 내용을 종합하여 답변합니다.")
            st.write("**분석 대상 사업 리스트:**")
            for p_name in target_rows['사업명']:
                st.write(f"- {p_name}")
        else:
            # 단일 사업 정보 표시
            st.info(f"💰 예산: {target_row['사업 금액']} / 📅 마감: {target_row['입찰 참여 마감일']}")
            with st.expander("📄 원본 텍스트 보기"):
                st.write(combined_context)

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

            with st.chat_message("assistant"):
                status = st.status("🧠 데이터를 종합 분석 중입니다..." if is_multi else "🔍 RFP 분석 중...")
                
                try:
                    # [핵심] 프롬프트 빌더 호출
                    # 다중 문서일 때는 'IT_정보화' 카테고리의 기본 페르소나를 사용하도록 설정
                    final_messages = builder.build_messages(
                        category=selected_d1,
                        title=selected_project,
                        context=combined_context,
                        history=st.session_state.messages[:-1],
                        query=query
                    )

                    with st.expander("🛠️ Debug: 조립된 컨텍스트 확인"):
                        st.write(f"컨텍스트 길이: {len(combined_context)}자")
                        st.json(final_messages)

                    # LLM 호출
                    answer = get_llm_response_safe(final_messages, client=client)
                    
                    status.update(label="✅ 분석 완료!", state="complete", expanded=False)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    status.update(label="❌ 오류 발생", state="error")
                    st.error(f"오류 내용: {str(e)}")

if __name__ == "__main__":
    main()