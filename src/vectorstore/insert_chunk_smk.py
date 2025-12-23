#==================================================================
# 프로그램명: insert_chunk.py
# 폴더 위치    : src/vectorstore/insert_chunk.py
# 프로그램 설명: json형태의 chunk및 메타데이터 Supabase vector table에 insert
# 작성자 : 2025-12-22 서민경, parent_section, related_section, depth1,depth2 추후 다른 json 규격에 맞추어 수정
#==================================================================

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from pathlib import Path

# -----------------------------------
# 1. 프로젝트 루트 기준으로 .env 로드
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# -----------------------------------
# 2. 환경변수
# -----------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# -----------------------------------
# 3. 클라이언트 생성
# -----------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------
# 4. embedding 함수
# -----------------------------------
def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# -----------------------------------
# 5. insert 대상 JSON 로드
# -----------------------------------
CHUNKS_JSON_PATH = BASE_DIR / "src" / "dataset" / "chunks_one_smk_pdf.json"
with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------------
# 6~7. embedding 생성 + Supabase insert
# -----------------------------------
for chunk in chunks:
    embedding_input = f"""
사업명: {chunk['project_name']}

내용:
{chunk['text']}
""".strip()

    embedding = embed_text(embedding_input)

    result = supabase.table("documents_chunks").insert({
        "chunk_id": chunk["chunk_id"],
        "pages": chunk["pages"],

        "announcement_id": chunk.get("announcement_id"),
        "announcement_round": chunk.get("announcement_round"),
        "project_name": chunk["project_name"],
        "project_budget": (int(chunk["project_budget"]) if chunk.get("project_budget") is not None else None),
        "ordering_agency": chunk.get("ordering_agency"),
        "published_at": chunk.get("published_at"),
        "bid_start_at": chunk.get("bid_start_at"),
        "bid_end_at": chunk.get("bid_end_at"),

        "text": chunk["text"],
        "embedding": embedding,
        "source_file": chunk["source_file"],
        "file_type": chunk["file_type"],
        "length": chunk["length"],

        "content_type": chunk["metadata"]["content_type"],
        "chunk_index": chunk["metadata"]["chunk_index"],
 
    }).execute()

    print("✅ Insert result")
    print(result.data)