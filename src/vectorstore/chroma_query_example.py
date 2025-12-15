from vectorstore.chroma_client import get_chroma_collection
from processing.embedder import embed_text  # bge-m3 임베더
import pprint

collection = get_chroma_collection()


def query_chroma(query_text, where=None, n_results=5):
    # 1) 쿼리를 직접 bge-m3 으로 임베딩
    query_embedding = embed_text(query_text)

    # 2) Chroma query_embeddings 사용
    results = collection.query(
        query_embeddings=[query_embedding],
        where=where,
        n_results=n_results,
    )
    return results


def get_all():
    """전체 데이터 조회"""
    return collection.get(include=["documents", "metadatas"])


if __name__ == "__main__":
    res = query_chroma(
        query_text="클라우드 사업",
        where={"ordering_agency": "산업부"},
    )
    print("조건 검색:")
    pprint.pp(res)

    all_docs = get_all()
    print("\n전체 데이터:")
    pprint.pp(all_docs)
