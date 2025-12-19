from langchain_core.runnables import RunnableLambda
from generation.llm import get_llm
from prompt.loader import load_prompt

PROMPT = load_prompt("rag_qa", "1.0.0")


def _answer_logic(inputs: dict):
    question = inputs["question"]
    docs = inputs["top_docs"]

    context_blocks = []
    for d in docs:
        m = d["meta"]
        context_blocks.append(
            f"chunk_id={d['chunk_id']} | page={m.get('page')} | section={m.get('section')}\n"
            f"{d['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    llm = get_llm()
    prompt = PROMPT.format(
        question=question,
        context=context,
    )

    answer = llm.invoke(prompt).content

    return {
        "question": question,
        "context": context,  # 🔥 LangSmith에 남음
        "answer": answer,
    }


answer_generator = RunnableLambda(_answer_logic).with_config(
    run_name="AnswerGenerator",
    tags=["generation", "answer"],
)
