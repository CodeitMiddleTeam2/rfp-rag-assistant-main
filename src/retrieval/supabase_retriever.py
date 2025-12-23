#==============================================
# 프로그램명: supabase_retriever.py
# 폴더위치: src/retrieval/supabase_retriever.py
# 프로그램 설명:
#   - Supabase(pgvector)에서 Dense 검색(RPC 호출)로 후보 문서를 가져온다.
#   - 필터가 없으면 LIKE '%'로 처리하여 전체 검색.
#   - 최종 확정안: Dense + Rerank(bge_rerank_m3_ko)
# 작성이력: 2025.12.23 정예진 정리
#==============================================

from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from supabase import create_client  # pip install supabase
import os
import src.config.retriever_setting as rs
from src.retrieval.retrievers import wrap_with_reranker

class SupabaseDenseRetriever:
    """
    Supabase RPC(match_rag_chunks)를 호출해서 벡터 유사도 Top_N을 가져오는 Retriever.
    - invoke(query) 지원 (LangChain retriever처럼 사용 가능)
    - 반환: List[Document]
    """
    def __init__(self,
                 *,
                 embeddings,    # get_embeddings()가 만든 embeddings 객체 (embed_query 사용)
                 rpc_name: str = "match_rag_chunks",
                 default_filters: Optional[Dict[str,Any]]=None):
        # Supabase 연결 정보는 환경변수로 받는게 안전(코드/깃에 노출 금지)
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")   # 보통 서비스를 또는 RLS 허용 키
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_KEY 환경변수가 없습니다."
                "Python Run Config 또는 .env로 설정하세요."
            )
        self.sb = create_client(self.supabase_url, self.supabase_key)

        self.embeddings = embeddings
        self.rpc_name = rpc_name

        # 기본 필터(없으면 전체 검색)
        self.default_filters = default_filters or {}
