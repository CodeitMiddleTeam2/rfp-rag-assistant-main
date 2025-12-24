import os
import streamlit as st
from supabase import create_client, Client
from openai import OpenAI

#==============================================
# 프로그램명: supabase_manager.py
# 폴더위치: src/generation/supabase_manager.py
# 프로그램 설명: supabase DB 를 웹 데모에 연동하기 위한 클래스
# 작성이력: 25.12.23 한상준 최초 작성
#===============================================

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

    def get_embedding(self, text: str):
        """질문을 벡터로 변환 (데이터 팀이 사용한 모델과 일치해야 함!)"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def similarity_search(self, query: str, filters: dict = None, top_k: int = 5):
        """
        벡터 검색을 수행합니다.
        filters: {'depth_1': 'IT', 'project_name': '...'} 등의 메타데이터 필터
        """
        try:
            query_vector = self.get_embedding(query)
            
            # Supabase RPC 호출 (데이터 팀이 만든 함수명이 'match_documents'라고 가정)
            # RPC 파라미터 구조는 데이터 팀 설정에 따라 다를 수 있습니다.
            rpc_params = {
                "query_embedding": query_vector,
                "match_threshold": 0.5, # 유사도 임계값
                "match_count": top_k,
                # 필터가 있다면 전달 (구현 방식에 따라 다름, 여기선 예시)
                # "filter": filters 
            }
            
            # 메타데이터 필터링이 포함된 RPC를 호출하거나, 
            # 혹은 Python 레벨에서 post-filtering을 할 수도 있습니다.
            # 여기서는 가장 일반적인 RPC 호출 예시입니다.
            response = self.supabase.rpc("match_documents", rpc_params).execute()
            
            return response.data
            
        except Exception as e:
            st.error(f"❌ Supabase 검색 실패: {e}")
            return []

    def format_docs(self, docs):
        """검색된 문서들을 LLM에 넣기 좋게 텍스트로 합칩니다."""
        context = ""
        for doc in docs:
            # 데이터 팀이 저장한 컬럼명(content, chunk 등) 확인 필요
            content = doc.get("content", "") 
            meta = doc.get("metadata", {}) # 혹은 개별 컬럼
            source = meta.get("사업명", "Unknown")
            
            context += f"### 출처: {source}\n내용: {content}\n\n"
        return context