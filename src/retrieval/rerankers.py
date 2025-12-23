#==============================================
# 프로그램명: rerankers.py
# 폴더위치: src/retrieval/rerankers.py
# 프로그램 설명:
#   - "dense 결과"를 CrossEncoder로 rerank 해서 더 정확한 Top-K를 뽑는다.
#   - 최종 확정: dragonkue/bge-reranker-v2-m3-ko (한국어 최적화)
# 작성이력: 2025.12.19 정예진 최초 작성
#==============================================

from __future__ import annotations

from typing import List, Any
from langchain_core.documents import Document
import torch
import src.config.retriever_setting as rs
from sentence_transformers import CrossEncoder  # pip install -U sentence-transformers

RERANKER_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"

# 프로세스 내 캐시 (매번 다운로드/로드 방지)
CE_MODEL = None
CE_NAME = None

def load_cross_encoder(model_name: str):
    """
    sentence-transformers CrossEncoder 로더
    - reranker는 (qeury, doc) 쌍을 직접 입력으로 받아 점수를 출력하는 "재정렬" 모델
    - 한 번 로드하면 메모리에 캐시해두고 재사용
    """
    global CE_MODEL, CE_NAME
    if CE_MODEL is not None and CE_NAME == model_name:
        return CE_MODEL

    CE_MODEL = CrossEncoder(model_name, default_activation_function=torch.nn.Sigmoid())
    CE_NAME = model_name
    return CE_MODEL

class RerankWrapper:
    """
    base retriever 결과를 rerank 후 최종 final_k개 반환하는 래퍼
    - candidate_k: rerank할 후보 개수(초기 검색은 더 크게)
    - final_k : rerank 후 최종 반환 개수(LLM에 넣을 Top_K)
    """
    def __init__(self, base_retriever: Any, *, candidate_k:int, final_k:int):
        self.base = base_retriever
        self.candidate_k = int(candidate_k)
        self.final_k = int(final_k)

    def invoke(self, query:str) -> List[Document]:
        # 후보들 먼저 크게 가져온다 (중요: base_retriever 자체도 candidate_k로 뽑게 세팅되어야 함)
        docs = self.base.invoke(query)[: self.candidate_k]
        if not docs:
            return docs
        # CrossEncoder로 (query, doc) 점수 계산 후 재정렬
        ce = load_cross_encoder(RERANKER_MODEL_NAME)
        pairs = [(query, d.page_content) for d in docs]
        scores = ce.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)

        # 최종 final_k개 반환
        return [d for d, _ in ranked[: self.final_k]]

def wrap_with_reranker(base_retriever:Any, *, candidate_k: int | None = None, final_k:int | None=None):
    """
    최종 확정 파이프라인: 무조건 rerank 적용
    - candidate_k / final_k는 retriever_setting 값을 기본으로 사용
    """
    ck = int(candidate_k if candidate_k is not None else rs.RERANK_CANDIDATE_K)
    fk = int(final_k if final_k is not None else rs.TOP_K)  # 평가/운영 기준을 TOP_K로 맞춤
    ck = max(ck,fk)

    return RerankWrapper(base_retriever, candidate_k=ck, final_k=fk)



