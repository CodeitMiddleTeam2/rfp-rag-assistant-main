#==============================================
# 프로그램명: embedding_registry.py
# 폴더위치: src/config/embedding_registry.py
# 프로그램 설명: 임베딩모델키 -> 생성함수 자동매핑
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class EmbeddingSpec:
    """
    임베딩 모델을 생성하기 위한 스펙 단위
    - provider: "hf" 또는 "openai"
    - model_name: HF repo id / OpenAI model id
    - kwargs: provider별 추가 파라미터(필요할 때만)
    """
    key: str
    provider: str
    model_name: str
    kwargs: Dict[str, Any]

# 여기서 키만 추가하면, 코드 전체에서 자동으로 모델 선택 가능
REGISTRY: Dict[str, EmbeddingSpec] = {
    # Hugging Face
    "bge_m3": EmbeddingSpec(
        key = "bge_m3",
        provider = "hf",
        model_name = "BAAI/bge-m3",
        kwargs={}
    ),
    "koe5": EmbeddingSpec(
        key = "koe5",
        provider="hf",
        model_name="nlpai-lab/KoE5",
        kwargs={}
    ),
    "kure": EmbeddingSpec(
        key="kure",
        provider="hf",
        model_name="nlpai-lab/KURE-v1",
        kwargs={}
    ),
    # OpenAI
    "open_3_small": EmbeddingSpec(
        key="openai_3_small",
        provider="openai",
        model_name="text-embedding-3-large",
        kwargs={}
    )
}

def get_spec(key:str) -> EmbeddingSpec:
    """
    키로 스펙을 가져오는 함수
    - embedding_setting.py에서 지정한 EMBEDDING_KEY가 여기 REGISTRY에 존재해야함
    """
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown EMBEDDING_KEY='{key}'."
            f"Available keys: {list(REGISTRY.keys())}"
        )
    return REGISTRY[key]