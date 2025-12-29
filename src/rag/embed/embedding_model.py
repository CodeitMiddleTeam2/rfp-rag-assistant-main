#==============================================
# 프로그램명: embedding_model.py
# 폴더위치: ./src/rag/embed/embedding_model.py
# 프로그램 설명: 임베딩 모델 클래스
# 작성이력: 2025.12.26 정예진 최초 작성
# 25.12.29 한상준 filter_source 인자 추가 및 전달, LangSmith 추적 추가
# 25.12.29 db.query 호출할 때 변경된 인자(match_threshold) 전달
#==============================================

import openai
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from src.rag.db import Supabase


class EmbeddingModel:
    def __init__(self, model_name: str):
        super().__init__()
        self.model = OpenAIEmbeddings(model=model_name)
        self.db = Supabase()

    @traceable(run_type="retriever", name="Supabase_Dense_Search")
    def search(self, query:str, result_count:int=10, threshold:float=0.3) -> list[dict]:
        if query == "":
            raise Exception("질문이 비어있습니다.")
        try:
            embedded_query = self.model.embed_query(query)

            return self.db.query(embedded_query, result_count, match_threshold=threshold)
        except openai.NotFoundError as e:
            raise Exception(f"[embedding_model.py] 임베딩 모델 에러: {e}")


