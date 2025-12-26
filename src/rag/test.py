#==============================================
# 프로그램명: test.py
# 폴더위치: ./src/rag/test.py
# 프로그램 설명:
# 사용자 질문인 Qurey가 들어왔을 때 임베딩 모델이 텍스트 임베딩 후 Supabase DB에서 유사도 검색을 수행하고, Rerank 모델이 재정렬하여 최종 쿼리를 생산합니다.
# 작성이력: 2025.12.26 정예진 최초 작성
#==============================================


from dotenv import load_dotenv

from src.rag.embed.embedding_model import EmbeddingModel
from src.rag.rerank.rerank_model import RerankModel

load_dotenv()

QUERY = "대학 관련 사업의 공고번호와 사업명은 무엇인가요?"

embedding_model = EmbeddingModel("text-embedding-3-small")
rerank_model = RerankModel("dragonkue/bge-reranker-v2-m3-ko")

retrieval_results = embedding_model.search(QUERY, result_count=10) # 유사도 검색된 결과
results = rerank_model.rerank(QUERY, retrieval_results, top_k=10)
print(type(results))
print(results.content)  # 랭체인 HumanMessage 객체

