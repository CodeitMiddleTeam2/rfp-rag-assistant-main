# ==============================================
# 프로그램명: db.py
# 폴더위치: ./src/rag/db.py
# 프로그램 설명: supabase 기능 관련 클래스
#   - query() : 임베딩된 사용자 쿼리를 받고, 유사도 계산하여 return하는 메서드
# 작성이력: 2025.12.26 정예진 최초 작성
# 25.12.29 한상준 query 함수에 filter_source 인자 추가
# 25.12.29 supabase DB 검색 함수를 match_rag_chunks >> match_documents_chunks_structural_vector 함수로 변경
# ==============================================
import os

from pydantic import Json
from supabase import create_client

class Supabase:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.client = create_client(url, key)

    def insert(self, data:Json) -> bool:
        # 전처리 json db insert 작업 여기서 해주세요
        print("insert 진행합니다.")
        return True

    def query(self, embedded_query:list[float], result_count:int, match_threshold:float=0.3) -> list[dict]:
        try:
            return self.client.rpc(
                "match_documents_chunks_structural_vector",
                {
                    "query_embedding": embedded_query,
                    "match_threshold": match_threshold,
                    "match_count": result_count
                }).execute().data
        except Exception as e:
            raise Exception(f"[db.py] match_documents_chunks_structural_vector 실행 실패: {e}")