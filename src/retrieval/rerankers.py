#==============================================
# 프로그램명: rerankers.py
# 폴더위치: src/retrieval/rerankers.py
# 프로그램 설명:
#   - RERANKER_KEY 설정에 따라 rerank(재정렬) 적용/미적용을 결정
# 작성이력: 2025.12.19 정예진 최초 작성
#==============================================

from __future__ import annotations

import importlib
from typing import Any

from src.config.retriever_setting import RERANKER_KEY, RERANK_TOP_N


def _import_any(candidates: list[str]) -> Any:
    """후보 import 경로를 순서대로 시도해서 성공하는 첫 번째를 반환."""
    last_err = None
    for path in candidates:
        try:
            module_path, attr = path.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, attr)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "필요한 클래스를 import 하지 못했습니다.\n"
        f"시도한 후보: {candidates}\n"
        f"마지막 에러: {last_err}"
    )


def wrap_with_reranker_if_needed(base_retriever):
    key = (RERANKER_KEY or "none").strip()
    if key == "none":
        return base_retriever

    # LangChain 쪽 클래스들: 버전에 따라 위치가 바뀌어서 후보를 다 시도
    ContextualCompressionRetriever = _import_any([
        "langchain.retrievers.contextual_compression.ContextualCompressionRetriever",
        "langchain.retrievers.ContextualCompressionRetriever",
    ])

    CrossEncoderReranker = _import_any([
        "langchain.retrievers.document_compressors.CrossEncoderReranker",
        "langchain.retrievers.document_compressors.cross_encoder.CrossEncoderReranker",
    ])

    # HF CrossEncoder: community 경로가 버전별로 달라질 수 있어 후보 처리
    HuggingFaceCrossEncoder = _import_any([
        "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
        "langchain_community.cross_encoders.huggingface.HuggingFaceCrossEncoder",
    ])

    # reranker 모델 선택
    if key == "bge_rerank_m3":
        model_name = "BAAI/bge-reranker-v2-m3"
    elif key == "bge_rerank_m3_ko":
        model_name = "dragonkue/bge-reranker-v2-m3-ko"
    else:
        raise ValueError(f"Unknown RERANKER_KEY='{key}' (use none/bge_rerank_m3/bge_rerank_m3_ko)")

    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=RERANK_TOP_N)

    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

