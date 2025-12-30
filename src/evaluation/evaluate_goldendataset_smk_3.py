#==================================================================
# 프로그램명: evaluate_goldendataset_smk.py
# 폴더 위치    : src/evaluation/evaluate_goldendataset_smk.py
# 프로그램 설명: golden dataset으로 키워드/백터 조회 결과 LLM질의를 한 결과, 질의/응답/context/ground truth를 josn으로 저장
#             - input: src/dataset/goldendataset.json
#             - output: src/dataset/ragas_inputs.json
# 작성이력 :       
#                 2025.12.18 오민경 최초작성
#                 2025.12.28 BM25 n-gram 검색 함수 반영(or연산), or 연산으로 timeout시 에러 무시 기능 추가(vector검색만 진행)
#==================================================================
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
import yaml
from supabase import create_client
from tqdm import tqdm

import torch
from sentence_transformers import CrossEncoder

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from postgrest.exceptions import APIError
from kiwipiepy import Kiwi

# ==================================================
# 0. 환경 로드 + LangSmith 설정
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rfp-rag-eval")


# ==================================================
# 1. Supabase / Embedding / LLM / Reranker
# ==================================================
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "dragonkue/bge-reranker-v2-m3-ko"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANKER_MODEL, device=device)


# ==================================================
# 2. Prompt YAML 로딩
# ==================================================
PROMPT_PATH = BASE_DIR / "src" / "prompts" / "ragas_template.yaml"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_yaml = yaml.safe_load(f)["prompt"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_yaml["system"]),
        (
            "human",
            prompt_yaml["instructions"]
            + "\n\n"
            + prompt_yaml["context_format"]
            + "\n\n"
            + prompt_yaml["user_prompt"]
            + "\n\n"
            + prompt_yaml["answer_guidelines"]
            + "\n\n"
            + prompt_yaml["output_format"],
        ),
    ]
)


# ==================================================
# 3. Retrieval RPC (Vector / BM25 OR n-gram)
# ==================================================
VECTOR_RPC = os.getenv(
    "VECTOR_RPC",
    "match_documents_chunks_smk2_vector"
)

# 🔥 핵심 수정: n-gram OR BM25 함수
BM25_RPC = os.getenv(
    "BM25_RPC",
    "match_documents_chunks_smk2_bm25_ngram"
)


def vector_search_fn(
    question: str,
    top_k: int = 20,
    threshold: float = 0.2
) -> List[Dict[str, Any]]:
    q_emb = embeddings.embed_query(question)

    res = supabase.rpc(
        VECTOR_RPC,
        {
            "query_embedding": q_emb,
            "match_threshold": threshold,
            "match_count": top_k,
        },
    ).execute()

    docs = res.data or []
    for d in docs:
        d["source"] = "vector"
        d["vector_score"] = float(d.get("score", 0.0))
    return docs


kiwi = Kiwi()

def extract_nouns_query(question: str) -> str:
    """
    Kiwi 기반 명사 추출 (BM25 n-gram 질의용)
    - NN*, NNP, NNB 포함
    - 1글자 토큰 제거
    """
    if not question:
        return ""

    result = kiwi.analyze(question)
    if not result:
        return ""

    nouns = [
        token.form
        for token in result[0][0]
        if token.tag.startswith("NN") and len(token.form) > 1
    ]

    return " ".join(nouns)


def bm25_search_fn(
    question: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    try:
        noun_query = extract_nouns_query(question)

        if not noun_query.strip():
            return []

        res = supabase.rpc(
            BM25_RPC,
            {
                "query": noun_query,
                "match_count": top_k,
            },
        ).execute()

        docs = res.data or []
        for d in docs:
            d["source"] = "bm25"
            d["bm25_score"] = float(d.get("score", 0.0))
            d["bm25_query"] = noun_query
        return docs

    except APIError as e:
        print(f"⚠️ BM25 skipped (APIError): {str(e)}")
        return []

    except Exception as e:
        print(f"⚠️ BM25 unexpected error → skip: {repr(e)}")
        return []


def hybrid_merge(
    vector_docs: List[Dict[str, Any]],
    bm25_docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    chunk_id 기준 병합
    - vector / bm25 / hybrid 명확히 구분
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for d in vector_docs:
        key = str(d["chunk_id"])
        merged[key] = d

    for d in bm25_docs:
        key = str(d["chunk_id"])
        if key in merged:
            merged[key]["source"] = "hybrid(vector+bm25)"
            merged[key]["bm25_score"] = d.get("bm25_score")
        else:
            merged[key] = d

    return list(merged.values())


# ==================================================
# 4. Rerank
# ==================================================
def bge_rerank(
    question: str,
    docs: List[Dict[str, Any]],
    k: int = 6
) -> List[Dict[str, Any]]:
    if not docs:
        return []

    pairs = []
    kept_docs = []

    for d in docs:
        text = d.get("text", "").strip()
        if not text:
            continue
        pairs.append((question, text))
        kept_docs.append(d)

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    for d, s in zip(kept_docs, scores):
        d["rerank_score"] = float(s)

    kept_docs.sort(
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True
    )
    return kept_docs[:k]


# ==================================================
# 5. Context 생성
# ==================================================
def build_contexts(docs: List[Dict[str, Any]]) -> List[str]:
    contexts = []

    for d in docs:
        metadata = d.get("metadata") or {}

        contexts.append(
            f"""
[공고 정보]
- 공고 번호: {d.get("announcement_id")}
- 공고 차수: {d.get("announcement_round")}
- 공고명: {d.get("project_name")}
- 사업 금액: {d.get("project_budget")}
- 발주 기관: {d.get("ordering_agency")}
- 공개 일자: {d.get("published_at")}



[본문]
{d.get("text")}
""".strip()
        )

    return contexts


# ==================================================
# 6. LangSmith Runnable Pipeline
# ==================================================
def step_vector(x):
    tqdm.write("    [1/5] Vector 검색")
    return {**x, "vector_docs": vector_search_fn(x["question"])}


def step_bm25(x):
    tqdm.write("    [2/5] BM25 n-gram OR 검색")
    return {**x, "bm25_docs": bm25_search_fn(x["question"])}


def step_merge(x):
    tqdm.write("    [3/5] Hybrid merge")
    return {
        **x,
        "merged_docs": hybrid_merge(
            x["vector_docs"],
            x["bm25_docs"],
        ),
    }


def step_rerank(x):
    tqdm.write("    [4/5] BGE rerank")
    return {
        **x,
        "reranked_docs": bge_rerank(
            x["question"],
            x["merged_docs"],
        ),
    }


def step_answer(x):
    tqdm.write("    [5/5] LLM answer")
    contexts = build_contexts(x["reranked_docs"])
    messages = prompt.format_messages(
        question=x["question"],
        contexts="\n\n".join(contexts),
    )
    answer = llm.invoke(messages)
    return {**x, "contexts": contexts, "answer": answer.content}


rag_pipeline = (
    RunnableLambda(lambda q: {"question": q})
    | RunnableLambda(step_vector)
    | RunnableLambda(step_bm25)
    | RunnableLambda(step_merge)
    | RunnableLambda(step_rerank)
    | RunnableLambda(step_answer)
)


# ==================================================
# 7. 실행
# ==================================================
if __name__ == "__main__":
    GOLDEN_PATH = BASE_DIR / "src" / "dataset" / "goldendataset.json"
    OUTPUT_PATH = BASE_DIR / "src" / "dataset" / "ragas_inputs.json"

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    results = []

    for item in tqdm(golden_data):
        out = rag_pipeline.invoke(item["question"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "contexts": out["contexts"],
                "answer": out["answer"],
                "ground_truth": item.get("ground_truth", ""),
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 완료: {OUTPUT_PATH}")
