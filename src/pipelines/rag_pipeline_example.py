from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)

from retrieval.hybrid import hybrid_retriever
from generation.reranker import reranker
from generation.answer import answer_generator

# ─────────────────────────────────────────────
# RAG PIPELINE (LangSmith Trace FULL)
# ─────────────────────────────────────────────

rag_pipeline = (
    RunnableParallel(
        question=RunnablePassthrough(),
        where=lambda _: None,
    )
    | hybrid_retriever
    | reranker
    | answer_generator
).with_config(
    run_name="RAG_Pipeline",
    tags=["rag", "hybrid", "langsmith"],
)


def run(question: str):
    return rag_pipeline.invoke(question)
