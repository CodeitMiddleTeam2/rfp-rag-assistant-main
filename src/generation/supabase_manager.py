import os
import streamlit as st
from supabase import create_client, Client
from openai import OpenAI
from sentence_transformers import CrossEncoder
import numpy as np

#==============================================
# 프로그램명: supabase_manager.py
# 폴더위치: src/generation/supabase_manager.py
# 프로그램 설명: supabase DB 를 웹 데모에 연동하기 위한 클래스
# 작성이력: 25.12.23 한상준 최초 작성
# 25.12.24 rerank 추가
# 25.12.29 supabase 검색 메서드 업데이트
#===============================================
RERANKER_MODEL_ID = "BAAI/bge-reranker-m3-ko"

class SupabaseManager:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.url or not self.key:
            st.error("🚨 Supabase 환경변수가 설정되지 않았습니다 (.env 확인)")
            st.stop()
            
        self.supabase: Client = create_client(self.url, self.key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)

        self.reranker = self._load_reranker()

    @st.cache_resource
    def _load_reranker(_self):
        """
        Reranker 모델을 로컬 GPU 메모리에 로드합니다. (최초 1회만 실행)
        """
        try:
            # print(f"🚀 Reranker 로딩 중: {RERANKER_MODEL_ID}")
            return CrossEncoder(RERANKER_MODEL_ID, device="cuda", max_length=512)
        except Exception as e:
            st.warning(f"⚠️ Reranker 로드 실패 (CPU 모드로 전환): {e}")
            return CrossEncoder(RERANKER_MODEL_ID, device="cpu")

    def get_embedding(self, text: str):
        """질문을 벡터로 변환 (데이터 팀이 사용한 모델과 일치해야 함!)"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def search_and_rerank(self, query: str, selected_project: str = "%", final_top_k: int = 3):
        try:
            # 1. 질문을 벡터로 변환
            query_vector = self.get_embedding(query)
            
            # 2. match_rag_chunks 함수 호출 (필터 적용)
            # [작성 의도] 사이드바에서 선택한 사업명을 filter_source에 매핑합니다.
            rpc_params = {
                "query_embedding": query_vector,
                "match_count": 25, # Rerank를 위해 넉넉히 가져옴
                "filter_source": selected_project if selected_project else "%"
            }
            
            response = self.supabase.rpc("match_rag_chunks", rpc_params).execute()
            candidates = response.data

            if not candidates:
                return []

            # --- 2단계: Reranking (Local GPU) ---
            # Reranker 입력 형식: [[질문, 문서1], [질문, 문서2], ...]
            # 참고: 청크가 4500토큰이어도 Reranker는 앞부분(512토큰) 위주로 판단합니다.
            rerank_pairs = []
            for doc in candidates:
                content = doc.get("content", "") # 컬럼명 확인 필요 (text, content 등)
                rerank_pairs.append([query, content])

            # 점수 계산
            scores = self.reranker.predict(rerank_pairs)

            # 점수와 문서를 묶어서 정렬
            scored_docs = list(zip(candidates, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True) # 점수 높은 순 정렬

            # 상위 K개 추출
            final_results = [doc for doc, score in scored_docs[:final_top_k]]
            
            return final_results

        except Exception as e:
            st.error(f"❌ 검색 실패: {e}")
            return []

    def format_docs(self, docs):
        """LLM 입력용 포맷팅"""
        context = ""
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            meta = doc.get("metadata", {})
            source = meta.get("사업명", doc.get("사업명", "Unknown Doc"))
            
            context += f"### 문서 {i+1} (출처: {source})\n{content}\n\n"
        return context