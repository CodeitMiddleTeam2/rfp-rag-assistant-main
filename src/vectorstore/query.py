#==================================================================
# 프로그램명: query.py
# 폴더 위치    : src/vectorstore/query.py
# 프로그램 설명: Supabase에서 문서 청크를 검색하는 스크립트
#             - vector search: Supabase의 match_documents_chunks RPC 호출
# 작성이력 :       
#                 2025.12.19 오민경 최초작성
#==================================================================
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os

# --------------------------------------------------
# 1. 프로젝트 루트 기준 .env 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# --------------------------------------------------
# 2. 클라이언트 생성
# --------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------
# 3. Embedding 함수
# --------------------------------------------------
def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# --------------------------------------------------
# 4. Vector Search RPC 호출
# --------------------------------------------------
def query_documents(
    query: str,
    match_threshold: float = 0.6,
    match_count: int = 5,
):
    query_embedding = embed_text(query)

    response = supabase.rpc(
        "match_documents_chunks",
        {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
        }
    ).execute()

    return response.data or []

# --------------------------------------------------
# 5. 실행부
# --------------------------------------------------
if __name__ == "__main__":
    query_text = "체육특기자 경기기록 관리시스템 개발 사업 내용은 무엇인가요?"

    results = query_documents(
        query=query_text,
        match_threshold=0.6,
        match_count=5
    )

    print("\n🔍 Vector Search 결과\n")

    if not results:
        print("검색 결과가 없습니다.")
    else:
        for idx, row in enumerate(results, 1):
            print(f"[{idx}] similarity: {row['similarity']:.3f}")
            print(f"사업명: {row['project_name']}")
            print(f"내용 미리보기: {row['text'][:300]}...")
            print("-" * 60)
