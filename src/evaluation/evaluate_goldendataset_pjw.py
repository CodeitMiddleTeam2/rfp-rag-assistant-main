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


# ==================================================
# 0. 환경 로드 + LangSmith 설정
# ==================================================
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# LangSmith tracing (필요 env: LANGCHAIN_API_KEY)
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

# ✅ 요청: bge reranker m3 ko
# (검색 결과 기준) dragonkue/bge-reranker-v2-m3-ko :contentReference[oaicite:0]{index=0}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
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
# 3. Retrieval RPC (Vector / BM25)
# ==================================================
VECTOR_RPC = os.getenv("VECTOR_RPC", "match_documents_chunks_structural_vector")
BM25_RPC = os.getenv("BM25_RPC", "match_documents_chunks_structural_bm25")


def vector_search_fn(question: str, top_k: int = 20, threshold: float = 0.2) -> List[Dict[str, Any]]:
    """Supabase RPC로 vector 검색"""
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


def bm25_search_fn(question: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Supabase RPC로 FTS(BM25 유사) 검색"""
    res = supabase.rpc(
        BM25_RPC,
        {
            "query": question,
            "match_count": top_k,
        },
    ).execute()

    docs = res.data or []
    for d in docs:
        d["source"] = "bm25"
        d["bm25_score"] = float(d.get("score", 0.0))
    return docs


def hybrid_merge(vector_docs: List[Dict[str, Any]], bm25_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """chunk_id 기준 dedup merge"""
    merged: Dict[str, Dict[str, Any]] = {}

    for d in vector_docs:
        key = str(d.get("chunk_id"))
        merged[key] = d

    for d in bm25_docs:
        key = str(d.get("chunk_id"))
        if key in merged:
            # 둘 다 걸린 경우 source 표시 강화
            merged[key]["source"] = "hybrid(vector+bm25)"
            merged[key]["bm25_score"] = float(d.get("bm25_score", d.get("score", 0.0)))
        else:
            merged[key] = d

    return list(merged.values())


# ==================================================
# 4. Rerank
# ==================================================
def bge_rerank(question: str, docs: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    """CrossEncoder로 재정렬"""
    if not docs:
        return []

    pairs = []
    kept_docs = []
    for d in docs:
        txt = d.get("text") or ""
        if not txt.strip():
            continue
        pairs.append((question, txt))
        kept_docs.append(d)

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    for d, s in zip(kept_docs, scores):
        d["rerank_score"] = float(s)

    kept_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return kept_docs[:k]


# ==================================================
# 5. Context 생성 (요청: 한글 컬럼명 포함)
# ==================================================
def build_contexts(docs: List[Dict[str, Any]]) -> List[str]:
    contexts: List[str] = []

    for d in docs:
        # ✅ 핵심: metadata 안전하게 추출
        metadata = d.get("metadata") or {}

        parent_section = metadata.get("parent_section", "정보 없음")
        related_section = metadata.get("related_section", "정보 없음")

        contexts.append(
            f"""
[공고 정보]
- 공고 번호: {d.get("announcement_id")}
- 공고 차수: {d.get("announcement_round")}
- 공고명: {d.get("project_name")}
- 사업 금액: {d.get("project_budget")}
- 발주 기관: {d.get("ordering_agency")}
- 공개 일자: {d.get("published_at")}
- 입찰 참여 시작일: {d.get("bid_start_at")}
- 입찰 참여 마감일: {d.get("bid_end_at")}
- 파일명: {d.get("source_file")}
- 파일형식: {d.get("file_type")}
- 문서 길이(length): {d.get("length")}

[문서 구조 정보]
- 상위 섹션(parent_section): {parent_section}
- 연관 섹션(related_section): {related_section}

[검색 정보]
- 검색 소스: {d.get("source")}
- vector_score: {d.get("vector_score")}
- bm25_score: {d.get("bm25_score")}
- rerank_score: {d.get("rerank_score")}

[사업요약]
{d.get("text")}
""".strip()
        )

    return contexts


# ==================================================
# 6. LangSmith 단계별 Tracing을 위한 Runnable 구성
# ==================================================
def step_vector(x: Dict[str, Any]) -> Dict[str, Any]:
    q = x["question"]
    tqdm.write(f"    [1/5] Vector 검색 중...")
    vdocs = vector_search_fn(q, top_k=20, threshold=0.2)
    return {**x, "vector_docs": vdocs}


def step_bm25(x: Dict[str, Any]) -> Dict[str, Any]:
    q = x["question"]
    tqdm.write(f"    [2/5] BM25(유사) 키워드 검색 중...")
    kdocs = bm25_search_fn(q, top_k=20)
    return {**x, "bm25_docs": kdocs}


def step_merge(x: Dict[str, Any]) -> Dict[str, Any]:
    tqdm.write(f"    [3/5] Hybrid merge 중...")
    merged = hybrid_merge(x.get("vector_docs", []), x.get("bm25_docs", []))
    return {**x, "merged_docs": merged}


def step_rerank(x: Dict[str, Any]) -> Dict[str, Any]:
    q = x["question"]
    tqdm.write(f"    [4/5] BGE rerank 중... (model={RERANKER_MODEL}, device={device})")
    reranked = bge_rerank(q, x.get("merged_docs", []), k=6)
    return {**x, "reranked_docs": reranked}


def step_context_and_answer(x: Dict[str, Any]) -> Dict[str, Any]:
    q = x["question"]
    tqdm.write(f"    [5/5] LLM 답변 생성 중...")
    contexts = build_contexts(x.get("reranked_docs", []))
    ctx_text = "\n\n".join(contexts) if contexts else "관련 문서가 제공되지 않았습니다."

    messages = prompt.format_messages(question=q, contexts=ctx_text)
    answer = llm.invoke(messages)
    return {**x, "contexts": contexts, "answer": answer.content}


rag_pipeline = (
    RunnableLambda(lambda q: {"question": q}).with_config(run_name="00_User_Query")
    | RunnableLambda(step_vector).with_config(run_name="01_Vector_Retrieval")
    | RunnableLambda(step_bm25).with_config(run_name="02_BM25_Retrieval")
    | RunnableLambda(step_merge).with_config(run_name="03_Hybrid_Merge")
    | RunnableLambda(step_rerank).with_config(run_name="04_BGE_Reranking")
    | RunnableLambda(step_context_and_answer).with_config(run_name="05_Context_Prompt_LLM")
).with_config(run_name="RAG_Pipeline_Hybrid_BGE")


# ==================================================
# 7. 실행
# ==================================================
if __name__ == "__main__":
    GOLDEN_PATH = BASE_DIR / "src" / "dataset" / "goldendataset.json"
    OUTPUT_PATH = BASE_DIR / "src" / "dataset" / "ragas_inputs.json"

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"\n🚀 Hybrid(Vector+BM25) + BGE Reranker + LangSmith tracing 시작 (총 {len(golden_data)}개)\n")
    results = []

    for item in tqdm(golden_data, desc="Hybrid RAG tracing", unit="sample"):
        qid = item["id"]
        question = item["question"]

        tqdm.write(f"\n▶ [{qid}] 처리 시작")
        try:
            out = rag_pipeline.invoke(
                question,
                config={"tags": [qid]},  # LangSmith에서 id로 필터링 가능
            )

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "contexts": out.get("contexts", []),
                    "answer": out.get("answer", ""),
                    "ground_truth": item.get("ground_truth", ""),
                }
            )

            tqdm.write(f"✔ [{qid}] 완료")

        except Exception as e:
            tqdm.write(f"✖ [{qid}] 실패: {repr(e)}")
            # 실패 샘플도 남기고 싶으면 아래처럼 저장
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "contexts": [],
                    "answer": "",
                    "ground_truth": item.get("ground_truth", ""),
                    "error": repr(e),
                }
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료: Hybrid RAG + BGE reranker + RAGAS 입력 생성")
    print(f"📄 결과 파일: {OUTPUT_PATH}")
