#==============================================
# 프로그램명: rerank_model.py
# 폴더위치: ./src/rag/rerank/embedding_model.py
# 프로그램 설명: 리랭크 모델 클래스
# 작성이력: 2025.12.26 정예진 최초 작성
# 25.12.29 한상준 LangSmith 추적 추가
#==============================================
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
from langsmith import traceable

class RerankModel:
    def __init__(self, model_name:str):
        super().__init__()
        try:
            self.model = CrossEncoder(model_name)
        except OSError:
            raise Exception("[rerank_model.py] rerank 모델 이름이 존재하지 않습니다.")

    @traceable(run_type="chain", name="BGE_Reranking")
    def rerank(self, query:str, retrieval_results:list[dict], top_k:int=0):
        if retrieval_results is None:
            raise Exception("[rerank_model.py] 벡터DB에서 유사도 검색한 결과가 존재하지 않습니다.")
        if len(retrieval_results) < top_k:
            top_k = len(retrieval_results)
        pairs = [(query, r["content"]) for r in retrieval_results]
        scores = self.model.predict(pairs, batch_size=16)
        for r, s in zip(retrieval_results, scores):
            r["rerank_score"] = float(s)
        retrieval_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return self._make_human_message(retrieval_results[:top_k], query)

    def _make_human_message(self, retrieval_results: list[dict], query: str) -> HumanMessage:
        context = []
        for i, c in enumerate(retrieval_results, start=1):
            content = c["content"]
            meta = c["metadata"]
            rerank_score = c["rerank_score"]
            dense_score = c["score"]

            context.append(
                f"[chunk:{i}] rerank_score:{rerank_score} dense_score:{dense_score} meta:{meta}\ncontent:{content}\n\n")
        return HumanMessage(content=f"[QUESTION]:{query}\n[CONTEXT]:{context}\n[INSTRUCTIONS]:CONTEXT에 있는 내용으로만 답할것")
