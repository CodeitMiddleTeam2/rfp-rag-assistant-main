#==============================================
# 프로그램명: embedding_model.py
# 폴더위치: ./src/rag/embed/embedding_model.py
# 프로그램 설명: 임베딩 모델 클래스
# 작성이력: 2025.12.26 정예진 최초 작성
#==============================================

import openai
from langchain_openai import OpenAIEmbeddings

from src.rag.db import Supabase


class EmbeddingModel:
    def __init__(self, model_name: str):
        super().__init__()
        self.model = OpenAIEmbeddings(model=model_name)
        self.db = Supabase()

    def search(self, query:str, result_count:int=10) -> list[dict]:
        if query == "":
            raise Exception("질문이 비어있습니다.")
        try:
            embedded_query = self.model.embed_query(query)
            return self.db.query(embedded_query, result_count)
        except openai.NotFoundError as e:
            raise Exception(f"[embedding_model.py] 임베딩 모델 이름이 존재하지 않습니다.")


