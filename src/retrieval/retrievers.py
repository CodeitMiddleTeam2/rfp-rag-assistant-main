#==============================================
# 프로그램명: retrievers.py
# 폴더위치: src/retrieval/retrievers.py
# 프로그램 설명:최종 확정: Dense(Vector) 검색 + 한국어 Reranker 적용
# 작성이력: 2025.12.18 정예진 최초 작성 / 2025.12.23 수정
#==============================================


from __future__ import annotations

from langchain_core.documents import Document
from typing import List, Any
import src.config.retriever_setting as rs
from src.retrieval.rerankers import wrap_with_reranker

def build_local_retriever(*, vectorstore, docs: List[Document] | None = None) -> Any:
    """
    최종 Retriever 생성
    - 1차 후보 검색: vectorstore(dense)에서 candidate_k 만큼 가져옴
    - 2차 재정렬   : bge_rerank_m3_ko로 rerank
    - 최종 반환    : TOP_K 개
    """
    candidate_k = int(rs.RERANK_CANDIDATE_K)
    final_k = int(rs.TOP_K)

    # rerank는 후보가 많을수록 좋아서, Dense 1차 검색 k를 candidate_k로 둔다.
    base = vectorstore.as_retriever(search_kwargs={"k": candidate_k})

    # rerank 적용(후보 candidate_k -> 최종 final_k)
    return wrap_with_reranker(base, candidate_k=candidate_k, final_k=final_k)