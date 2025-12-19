#==============================================
# 프로그램명: build_index_run.py
# 폴더위치: src/pipelines/build_index_run.py
# 프로그램 설명: "JSON 로딩 → 임베딩 → FAISS 인덱싱 → 간단 검색 테스트” 실행 스크립트
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

import json
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config.embedding_setting import (
EMBEDDING_KEY,
NORMALIZE_EMBEDDINGS,
DEVICE,
JSON_PATH,
ADD_TYPE_PREFIX
)
from src.config.retriever_setting import TOP_K
from src.processing.embeddings import get_embeddings
from src.config.embedding_registry import get_spec
from src.retrieval.retrievers import build_local_retriever

def load_team_json(json_path: str, add_type_prefix: bool = False) -> List[Document]:
    """
    전처리 JSON(list)을 LangChain Document 리스트로 변환
    JSON record 예상 구조:
    {
    "system_id": "...",
    "content": "...",   # 1000자 단위 chunk
    "metadata": {...}   # source, type, chunk_index 등
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for rec in data:
        system_id = rec.get("system_id") or ""
        content = rec.get("content", "") or ""
        metadata = rec.get("metadata", {}) or {}

        # 추적/평가/디버깅에 유용: system_id는 metadata에도 같이 넣어두기
        metadata["system_id"] = system_id

        # type을 본문 앞에 붙여서 검색 성능/의미 구분에 도움될 수 있음
        #예: [header]..., [table]...
        if add_type_prefix:
            t = metadata.get("type")
            if t:
                content = f"[{t}] {content}"
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def quick_sanity_check(retriever, query: str) -> None:
    """
    모델/인덱스가 제대로 매칭 됐는지 확인하는 디버그용
    임베딩 모델이 최종으로 정해지고 파이프라인이 안정화되면 지워도 되지만
    나중에 문서추가/전처리 변경/청크 규칙 변경되면 확인 가능 (안쓰면 주석 처리)
    """
    results = retriever.invoke(query)

    print(f"\n[Query] {query}")
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata
        preview = doc.page_content[:200].replace("\n"," ")
        print(f"\n--- Top {i} ---")
        print("system_id    :", meta.get("system_id"))
        print("type         :", meta.get("type"))
        print("source       :", meta.get("source"))
        print("chunk_index  :", meta.get("chunk_index"))
        print("content_prev :", preview)

def main():
    # 데이터 로딩
    docs = load_team_json(JSON_PATH, add_type_prefix=ADD_TYPE_PREFIX)
    print(f"Loaded docs: {len(docs)}")

    # 어떤 모델이 선택됐는지 출력(로그용)
    spec = get_spec(EMBEDDING_KEY)
    print(f"Embedding key  : {EMBEDDING_KEY}")
    print(f"Provider        : {spec.provider}")
    print(f"Model name      : {spec.model_name}")

    # 임베딩 객체 생성 (키 기반 자동 선택)
    embeddings = get_embeddings(
        EMBEDDING_KEY,
        normalize=NORMALIZE_EMBEDDINGS,
        device=DEVICE
    )
    # 벡터스토어 생성
    vectorstore = FAISS.from_documents(docs, embeddings)

    # RETRIVER_KEY에 따라 dense/bm25/hybrid 선택됨
    retriever = build_local_retriever(vectorstore=vectorstore, docs=docs)

    # 간단 retrieval 테스트
    quick_sanity_check(retriever, query="입찰 참여 마감일은 언제야?")

if __name__ == "__main__":
    main()

