#==================================================================
# 프로그램명: retrieve_bm25_ngrm.py
# 폴더 위치    : src/retrieve/retrieve_bm25_ngram.py
# 프로그램 설명: BM25 n-gram 검색 테스트 파익ㄹ
#             - input: src/dataset/goldendataset.json
#             - output: src/dataset/ragas_inputs.json
# 작성이력 :       
#                 2025.12.28 오민경 최초작성
#==================================================================


# ==================================================
# 0. 기본 import
# ==================================================
import os
from dotenv import load_dotenv
from supabase import create_client


# ==================================================
# 1. 환경 변수 로드 (.env)
# ==================================================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("❌ SUPABASE_URL 또는 SUPABASE_SERVICE_KEY가 없습니다.")


# ==================================================
# 2. Supabase Client 생성
# ==================================================
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
)

print("✅ Supabase client connected")


# ==================================================
# 3. BM25 N-gram RPC 호출
# ==================================================
BM25_RPC = "match_documents_chunks_structural_bm25_ngram"

query_text = "통합 정보시스템 구축 사전 컨설팅 용역"

top_k = 10

res = supabase.rpc(
    BM25_RPC,
    {
        "query": query_text,
        "match_count": top_k,
    },
).execute()


# ==================================================
# 4. 결과 출력
# ==================================================
docs = res.data or []

print(f"\n🔍 Query: {query_text}")
print(f"📄 Retrieved documents: {len(docs)}\n")

if not docs:
    print("⚠️ BM25 결과가 없습니다.")
else:
    # 첫 번째 결과만 미리보기
    first = docs[0]

    print("✅ First result preview")
    print("-" * 60)
    print(f"chunk_id        : {first.get('chunk_id')}")
    print(f"announcement_id : {first.get('announcement_id')}")
    print(f"project_name    : {first.get('project_name')}")
    print(f"ordering_agency : {first.get('ordering_agency')}")
    print(f"score           : {first.get('score')}")
    print(f"text (preview)  : {first.get('text', '')[:200]}...")
    print("-" * 60)
