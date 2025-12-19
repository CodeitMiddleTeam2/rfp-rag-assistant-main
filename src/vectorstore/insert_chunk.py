#==================================================================
# 프로그램명: insert_chunk.py
# 폴더 위치    : src/vectorstore/insert_chunk.py
# 프로그램 설명: json형태의 chunk및 메타데이터 Supabase vector table에 insert
#             - table명: documents_chunks
#             - embedding 모델: openai text-embedding-3-small
#             - embedding 입력: 사업명 + 내용
#             - 메타데이터: lang(ko), embedding_source(project_name + text)
#             - 환경변수: SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY
#             - 사용라이브러리: supabase, openai, python-dotenv
#             - 실행방법: python3 -m src.vectorstore.insert_chunk
#             - 실행결과: Supabase documents_chunks table에 데이터 insert
# 작성이력 :       
#                 2025.12.19 오민경 최초작성
#==================================================================

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import os
from pathlib import Path

# -----------------------------------
# 1. 프로젝트 루트 기준으로 .env 로드
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # rfp-rag-assistant-main
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
# 5. insert 대상 데이터
# -----------------------------------
chunk = {
  "chunk_id": "d1a4f0e2-1d7e-4b3e-9a4f-0001aabbcc03",
  "page": 1,
  "announcement_id": "20240812818",
  "announcement_round": 0,
  "project_name": "KUSF 체육특기자 경기기록 관리시스템 개발",
  "project_budget": 150000000,
  "ordering_agency": "한국대학스포츠협의회(KUSF)",
  "published_at": "2024-08-16T08:52:03",
  "bid_start_at": "2024-09-02T10:00:00",
  "bid_end_at": "2024-09-06T10:00:00",
  "text": "체육특기자 선발 과정에서 경기실적만으로 평가하는 기존 방식의 한계를 개선하기 위해, 경기력 평가지표를 산출하고 이를 관리할 수 있는 신규 시스템을 개발한다. 본 사업을 통해 학생선수 개인별 경기력 판단이 가능해지고 대입 전형의 공정성을 확보할 수 있을 것으로 기대된다.",
  "source_file": "(사)한국대학스포츠협의회_KUSF 체육특기자 경기기록 관리시스템 개발.hwp",
  "file_type": "hwp",
  "length": 382
}

# -----------------------------------
# 6. embedding 입력 구성 (project_name + text)
# -----------------------------------
embedding_input = f"""
사업명: {chunk['project_name']}

내용:
{chunk['text']}
""".strip()

embedding = embed_text(embedding_input)

# -----------------------------------
# 7. Supabase insert
# -----------------------------------
result = supabase.table("documents_chunks").insert({
    "chunk_id": chunk["chunk_id"],
    "page": chunk["page"],
    "announcement_id": chunk["announcement_id"],
    "announcement_round": chunk["announcement_round"],
    "project_name": chunk["project_name"],
    "project_budget": chunk["project_budget"],
    "ordering_agency": chunk["ordering_agency"],
    "published_at": chunk["published_at"],
    "bid_start_at": chunk["bid_start_at"],
    "bid_end_at": chunk["bid_end_at"],
    "text": chunk["text"],
    "embedding": embedding,
    "source_file": chunk["source_file"],
    "file_type": chunk["file_type"],
    "length": chunk["length"],
    "metadata": {
        "lang": "ko",
        "embedding_source": "project_name + text"
    }
}).execute()

print("✅ Insert result")
print(result.data)
