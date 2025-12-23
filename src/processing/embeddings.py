#==============================================
# 프로그램명: embeddings.py
# 폴더위치: src/processing/embeddings.py
# 프로그램 설명: 임베딩 객체를 생성하는 팩토리(단일 진입점)
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

"""
단일 진입점: get_embeddings(embedding_key, ...)
- embedding_key만 바꾸면
- registry에서 spec을 찾아
- provider별로 적절한 Embeddings 객체를 만들어 반환
"""

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.config.embedding_registry import get_spec

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_embeddings(embeddings_key: str, normalize: bool = True, device: str = "auto"):
    """
    Perameters
    - embedding_key: str
        registry에 등록된 키값
    - normalize: bool
        HF 임베딩에서 normalize_embeddings 여부 (cosine 유사도면 보통 True 추천)
    - device: str
        "auto" | "cuda" | "cpu"
    """
    spec = get_spec(embeddings_key)
    resolve_device = _resolve_device(device)

    # provider에 따라 다른 Embeddings 클래스를 생성
    if spec.provider == "hf":
        return HuggingFaceEmbeddings(
            model_name=spec.model_name,
            model_kwargs={"device": resolve_device},
            encode_kwargs={"normalize_embeddings": normalize},
            **spec.kwargs   # 옵션 필요할때 사용
        )
    if spec.provider == "openai":
        return OpenAIEmbeddings(
            model=spec.model_name,
            **spec.kwargs
        )
    raise ValueError(f"Unsupported provider: {spec.provider}")
