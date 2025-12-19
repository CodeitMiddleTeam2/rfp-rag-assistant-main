#==================================================================
# 프로그램명: query_hybrid_rerank.py
# 폴더 위치    : src/vectorstore/query_hybrid_rerank.py
# 프로그램 설명: 하이브리드 검색 및 리랭킹을 수행하는 스크립트
#             - vector search: Supabase의 match_documents_chunks RPC 호출
#             - keyword search: project_name 또는 text에 keyword 포함 여부 검색
#             - hybrid search: vector + keyword 검색 결과 병합 및 점수 조합
#             - rerank: CrossEncoder를 사용한 리랭킹
# 작성이력 :       
#                 2025.12.19 오민경 최초작성
#==================================================================

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder
from pathlib import Path
import os

# --------------------------------------------------
# 1. 환경 변수 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# 2. Embedding 함수
# --------------------------------------------------
def embed_text(text: str) -> list[float]:
    res = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

# --------------------------------------------------
# 3. Vector Search
# --------------------------------------------------
def vector_search(query: str, top_k: int = 8):
    query_embedding = embed_text(query)

    res = supabase.rpc(
        "match_documents_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": top_k
        }
    ).execute()

    return res.data or []

# --------------------------------------------------
# 4. Keyword Search
# --------------------------------------------------
def keyword_search(keyword: str, limit: int = 5):
    res = (
        supabase
        .table("documents_chunks")
        .select("chunk_id, project_name, text")
        .or_(
            f"project_name.ilike.%{keyword}%,text.ilike.%{keyword}%"
        )
        .limit(limit)
        .execute()
    )
    return res.data or []

# --------------------------------------------------
# 5. Hybrid Search
# --------------------------------------------------
def hybrid_search(query: str):
    keyword = "체육특기자"

    vector_results = vector_search(query)
    keyword_results = keyword_search(keyword)

    merged = {}

    for r in vector_results:
        merged[r["chunk_id"]] = {
            "chunk_id": r["chunk_id"],
            "project_name": r["project_name"],
            "text": r["text"],
            "score": r["similarity"] * 0.7,
            "source": "vector"
        }

    for r in keyword_results:
        if r["chunk_id"] in merged:
            merged[r["chunk_id"]]["score"] += 0.3
            merged[r["chunk_id"]]["source"] += "+keyword"
        else:
            merged[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "project_name": r["project_name"],
                "text": r["text"],
                "score": 0.5,
                "source": "keyword"
            }

    # Hybrid score 기준 상위 후보만 리랭커로 전달
    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:10]

# --------------------------------------------------
# 6. Reranker (Cross Encoder)
# --------------------------------------------------
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query: str, candidates: list, top_n: int = 3):
    pairs = [
        (query, f"{c['project_name']}\n{c['text']}")
        for c in candidates
    ]

    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

# --------------------------------------------------
# 7. 실행부
# --------------------------------------------------
if __name__ == "__main__":
    query_text = "체육특기자 경기기록 관리시스템 개발 사업 내용은?"

    # 1) Hybrid 검색
    hybrid_results = hybrid_search(query_text)

    # 2) Rerank
    final_results = rerank(query_text, hybrid_results, top_n=3)

    print("\n🔥 Hybrid + Rerank 결과\n")

    for i, r in enumerate(final_results, 1):
        print(f"[{i}] rerank_score={r['rerank_score']:.4f}")
        print(f"사업명: {r['project_name']}")
        print(f"출처: {r['source']}")
        print(f"내용 미리보기: {r['text'][:300]}...")
        print("-" * 70)
