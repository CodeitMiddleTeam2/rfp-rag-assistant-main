#==============================================
# 프로그램명: retrievers.py
# 폴더위치: src/retrieval/retrievers.py
# 프로그램 설명:
#   - RETRIEVER_KEY 설정에 따라 retriever 생성
#   - LOCAL 하이브리드 검색 모듈
#   - FAISS(벡터검색) + BM25(키워드 검색) + Hybrid 비교를 쉽게 하기 위한 파일
#   - Supabase(DB)와 무관하게 동작하도록 설계됨
# 작성이력: 2025.12.18 정예진 최초 작성
#==============================================

from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.config.retriever_setting import RETRIEVER_KEY, TOP_K, HYBRID_WEIGHTS
from src.retrieval.rerankers import wrap_with_reranker_if_needed

# 내부 유틸: 버전 차이/경로 차이로 import 실패하는 경우를 대비한 안전 import
"""
EnsembleReriever 같은 건 LangChain 버전/패키지 조합에 따라 import 경로가 달라짐
그래서 여러 경로를 순서대로 시도하는 안전 import
"""
def _import_bm25_retriever():
    """BM25Retriever는 보통 langchain_community.retrievers에 있음"""
    from langchain_community.retrievers import BM25Retriever
    return BM25Retriever

def _import_ensemble_retriever():
    # 가장 흔한 경로
    try:
        from langchain.retrievers import EnsembleRetriever
        return EnsembleRetriever
    except Exception:
        pass
    # 버전에 따라 이 경로가 필요할 수 있음
    try:
        from langchain.retrievers.ensemble import EnsembleRetriever
        return EnsembleRetriever
    except Exception:
        pass
    # 일부 환경에서 community로 빠진 경우도 대비
    try:
        from langchain_community.retrievers import EnsembleRetriever
        return EnsembleRetriever
    except Exception as e:
        raise ImportError(
            "EnsembleRetriever import 실패"
            "langchain / langchain-core / langchain-community 설치 상태와 버전을 확인하세요.\n"
            f"원인: {e}"
        )

# ------------------------------------------------------
# 핵심 함수: LOCAL retriever 생성
#-------------------------------------------------------
def build_local_retriever(*, vectorstore, docs: List[Document]):
    """
    설정(RETRIEVER_KEY)에 따라 retriever 객체를 생성해서 반환
    Parameters
    vectorstore: dense(벡터) 검색용 (예: FAISS)
    docs: BM25(키워드) 검색용 Document 리스트 (전처리/청킹된 chunk들)
    Returns
    retriever: .invoke(query) 로 검색 가능한 LangChain Retriever 객체
    """
    key = (RETRIEVER_KEY or "").strip()

    #  Dense Retriever: 벡터 유사도 검색 (Vector DB/FAISS)
    if key == "dense":
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        # 리랭크 적용 코드
        # base = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        # return wrap_with_reranker_if_needed(base)

    # BM25 Retriever: 키워드 검색 (Lexical)
    if key == "bm25":
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = TOP_K
        return bm25
        # 리랭크 적용 코드
        # base = BM25Retriever.from_documents(docs)
        # base.k = TOP_K
        # return wrap_with_reranker_if_needed(base)

    # Hybrid Retriever (BM25 + Dense -> RRF 결합)
    if key == "hybrid_rrf":
        BM25RetrieverCls  = _import_bm25_retriever()
        EnsembleRetrieverCls  = _import_ensemble_retriever()

        dense = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        bm25 = BM25RetrieverCls.from_documents(docs)
        bm25.k = TOP_K

        # weights는 retrievers=[bm25, dense] 순서에 맞춰야 함
        base = EnsembleRetrieverCls(
            retrievers=[bm25, dense],
            weights=HYBRID_WEIGHTS
        )
        return wrap_with_reranker_if_needed(base)
    # 잘못된 키 입력 시 즉시 알 수 있도록 에러
    raise ValueError(
        f"Unknown RETRIEVER_KEY='{key}'."
        "Use on or: 'dense', 'bm25', 'hybrid_rrf'."
    )


