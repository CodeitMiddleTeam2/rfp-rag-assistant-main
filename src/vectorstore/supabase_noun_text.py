#==================================================================
# 프로그램명: supanbase_noun_text.py
# 폴더 위치    : src/vectorestore/supabase_noun_text.py
# 프로그램 설명: supabase 키워드 검색시, 명사에 한하여 ngram처리가 되도록 하기 위해서 처리
#             - output: documents_chunks_structural테이블의 noun_text컬럼 
# 작성이력 :       
#                 2025.12.29 오민경 최초작성
#==================================================================

from dotenv import load_dotenv
from supabase import create_client
import os
from kiwipiepy import Kiwi
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


kiwi = Kiwi()

def extract_nouns(text: str) -> str:
    if not text:
        return ""
    result = kiwi.analyze(text)
    return " ".join(
        token.form
        for token in result[0][0]
        if token.tag.startswith("NN")  # NN, NNP, NNB
    )



BATCH = 1000

while True:
    rows = (
        supabase
        .table("documents_chunks_structural")
        .select(
            "chunk_id, announcement_id, project_name, ordering_agency, metadata"
        )
        .is_("noun_text", None)
        .limit(BATCH)
        .execute()
        .data
    )

    if not rows:
        break

    for r in rows:
        meta = r.get("metadata") or {}

        source_text = " ".join([
            r.get("announcement_id", ""),
            r.get("project_name", ""),
            r.get("ordering_agency", ""),
            meta.get("parent_section", ""),
            meta.get("related_section", ""),
        ])

        noun_text = extract_nouns(source_text)

        supabase.table("documents_chunks_structural") \
            .update({"noun_text": noun_text}) \
            .eq("chunk_id", r["chunk_id"]) \
            .execute()
