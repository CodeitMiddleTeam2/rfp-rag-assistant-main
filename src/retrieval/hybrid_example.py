from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import collect_runs

from retrieval.dense import dense_search
from retrieval.lexical import lexical_score
from config.settings import DENSE_WEIGHT, LEX_WEIGHT


def _hybrid_logic(inputs: dict):
    question = inputs["question"]
    where = inputs.get("where")

    dense_hits = dense_search(question, k=12, where=where)

    results = []
    for h in dense_hits:
        lex = lexical_score(question, h["text"])
        hybrid = DENSE_WEIGHT * h["dense_score"] + LEX_WEIGHT * lex
        h.update(
            {
                "lex_score": lex,
                "hybrid_score": hybrid,
            }
        )
        results.append(h)

    results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)

    return {
        "question": question,
        "retrieval_results": results[:10],  # LangSmith에 남김
    }


hybrid_retriever = RunnableLambda(_hybrid_logic).with_config(
    run_name="HybridRetriever",
    tags=["retrieval", "hybrid"],
)
