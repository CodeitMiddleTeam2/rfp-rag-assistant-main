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

    def search_and_rerank(self, query: str, filters: dict = None, initial_top_k: int = 15, final_top_k: int = 3):
        """
        [핵심 로직] Dense Search -> Reranking 파이프라인
        1. Dense: 벡터 유사도로 넉넉하게 15개 정도 가져옵니다.
        2. Rerank: 질문과 문서의 관계를 정밀 채점하여 상위 3개만 남깁니다.
        """
        try:
            # --- 1단계: Dense Search (Supabase RPC) ---
            query_vector = self.get_embedding(query)
            
            rpc_params = {
                "query_embedding": query_vector,
                "match_threshold": 0.3, # 1차 필터링 (너무 낮은건 제외)
                "match_count": initial_top_k,
                # "filter": filters # (DB RPC 함수가 필터를 지원하도록 구현되어 있어야 함)
            }
            
            # DB 호출 (함수명 'match_documents'는 데이터 팀 확인 필요)
            response = self.supabase.rpc("match_documents", rpc_params).execute()
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
            st.error(f"❌ 검색/Rerank 실패: {e}")
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