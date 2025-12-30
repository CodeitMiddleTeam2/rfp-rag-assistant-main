#==================================================================
# 프로그램명: insert_chunk_structural.py
# 폴더 위치    : src/vectorstore/insert_chunk_structural.py
# 프로그램 설명: 미리 임베딩 된 JSON 데이터를 Supabase에 insert
#             - table명: documents_chunks_structural
#             - embedding 모델: openai text-embedding-3-small
#             - embedding 입력: 사업명 + 내용
#             - 메타데이터: lang(ko), embedding_source(project_name + text)
#             - 환경변수: SUPABASE_URL, SUPABASE_SERVICE_KEY
#             - 사용라이브러리: supabase, python-dotenv, tqdm
#             - 실행방법: python3 -m src.vectorstore.insert_chunk_structural
#             - 실행결과: Supabase documents_chunks table에 데이터 insert
# 작성이력 :       
#                 2025.12.23 박지원 최초작성
#==================================================================

from supabase import create_client
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2] 
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

DATA_PATH = BASE_DIR / "src" / "dataset" / "real_final_ingest_data.json"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def main():

    if not DATA_PATH.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
        return

    print(f"데이터를 읽어오는 중: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    
    total_count = len(all_chunks)
    print(f"총 {total_count}개의 청크를 업로드합니다.")


    BATCH_SIZE = 100
    
    for i in tqdm(range(0, total_count, BATCH_SIZE), desc="Supabase 업로드 중"):
        batch = all_chunks[i : i + BATCH_SIZE]
        
        try:
            # 생성한 테이블명: documents_chunks_structural_pjw
            supabase.table("documents_chunks_structural_pjw").insert(batch).execute()
        except Exception as e:
            print(f"\n❌ [에러 발생] (Index {i}): {e}")
            continue

    print("\n✅ 모든 데이터가 Supabase에 성공적으로 Insert 되었습니다.")

if __name__ == "__main__":
    main()