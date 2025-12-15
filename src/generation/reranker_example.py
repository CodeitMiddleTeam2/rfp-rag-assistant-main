import json
from langchain_core.runnables import RunnableLambda
from generation.llm import get_llm


def _rerank_logic(inputs: dict):
    question = inputs["question"]
    docs = inputs["retrieval_results"]

    llm = get_llm()

    prompt = (
        "다음 문서들을 질문과의 관련성 기준으로 0~5점으로 평가하세요.\n"
        "JSON 배열만 출력하세요.\n\n"
        f"질문: {question}\n\n"
    )

    for d in docs:
        prompt += (
            f"chunk_id={d['chunk_id']} | section={d['meta'].get('section')}\n"
            f"{d['text']}\n\n"
        )

    try:
        resp = llm.invoke(prompt)
        scores = json.loads(resp.content)
        score_map = {s["chunk_id"]: s["score"] for s in scores}
    except Exception:
        score_map = {}

    reranked = sorted(
        docs,
        key=lambda d: score_map.get(d["chunk_id"], 0),
        reverse=True,
    )

    return {
        "question": question,
        "rerank_scores": score_map,  # 🔥 LangSmith에 남음
        "top_docs": reranked[:5],
    }


reranker = RunnableLambda(_rerank_logic).with_config(
    run_name="LLMReranker",
    tags=["generation", "rerank"],
)
