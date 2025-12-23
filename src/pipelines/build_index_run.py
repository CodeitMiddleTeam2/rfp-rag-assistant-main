#==============================================
# 프로그램명: build_index_run.py
# 폴더위치: src/pipelines/build_index_run.py
# 프로그램 설명: "JSON 로딩 → 임베딩 → FAISS 인덱싱 → 간단 검색 테스트” 실행 스크립트
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

import json
from typing import List, Dict, Tuple, Any
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config.embedding_setting import (
    EMBEDDING_KEY,
    NORMALIZE_EMBEDDINGS,
    DEVICE,
    JSON_PATH,
    ADD_TYPE_PREFIX,
    GOLDEN_JSON_PATH,
    MAX_DOCS,
)
import src.config.retriever_setting as rs

from src.config.retriever_setting import TOP_K
from src.processing.embeddings import get_embeddings
from src.config.embedding_registry import get_spec
from src.retrieval.retrievers import build_local_retriever


# LangSmith
import langsmith as ls
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

# =======================================================
# LangSmith 토글(원하는대로 True/False 변경)
# =======================================================
ENABLE_LANGSMITH = True       # 전체 tracing ON/OFF
TRACE_EVAL_CALLS = False      # 평가 루프(retriever.invoke 반복)를 trace에 남길지 디버깅 필요한 말만 True
TRACE_SANITY = False          # quick_sanity_check trace 남길지 UI로 결과 남기고 싶을 때 True`
LOG_FEEDBACK = True           # Recall/MRR을 Feedback score로도 저장할지(선택)

# =======================================================
# Retrieval 평가(Recall/MRR)용 유틸 함수들
# =======================================================
def project_root() -> Path:
    """프로젝트 루트 안전하게 잡기 위한 함수"""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> str:
    """경로 문자열을 '항상 열 수 있는 절대경로'로 바꿔주는 함수"""
    p = Path(path_str)
    return str(p) if p.is_absolute() else str(project_root() / p)


def normalize_text(s: str) -> str:
    """텍스트 정규화: 공백 정리 + 소문자화(영문만 영향)"""
    s = s.replace("u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def digits_only(s: str) -> str:
    """129,300,000원 -> 129300000 처럼 숫자만 남김"""
    return re.sub(r"[^\d]", "", s)


def load_golden_json(path_str: str) -> List[Dict[str, Any]]:
    path = resolve_path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("goldendataset.json은 JSON 리스트 형태여야 합니다.")
    return data


def build_gold_signals(ex: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    - contexts의 각 줄을 phrase 후보로 넣는다(너무 짧은건 제외)
    - contexts/ground_truth/answer에서 숫자 토큰을 뽑는다.
      (금액처럼 콤마가 있는 숫자도 digits_only로 추가)
    """
    phrases: List[str] = []
    numbers: List[str] = []

    contexts = ex.get("contexts", []) or []

    for line in contexts:
        line = str(line).strip()
        if not line:
            continue

        line = line.replace("\t", " ")
        norm = normalize_text(line)

        # 너무 짧은 건 근거로 쓰기 어려워 제외
        if len(norm) >= 12:
            phrases.append(norm)

        # 숫자 토큰(6자리 이상) 추출
        for m in re.findall(r"\b\d{6,}\b", line):
            numbers.append(m)

        # 콤마 포함 금액/숫자 -> digits_only로 보조
        d = digits_only(line)
        if len(d) >= 6:
            numbers.append(d)

    # contexts가 약하면 ground_truth/answer에서도 숫자만 보조로 추출
    for key in ["ground_truth", "answer"]:
        text = str(ex.get(key, "") or "")
        if text:
            d = digits_only(text)
            if len(d) >= 6:
                numbers.append(d)
            for m in re.findall(r"\b\d{6,}\b", text):
                numbers.append(m)

    def uniq(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "phrases": uniq(phrases),
        "numbers": uniq(numbers),
    }


def doc_contains_gold(doc: Document, gold: Dict[str, List[str]]) -> Tuple[bool, str]:
    """
    HIT 판정 규칙(혼합형):
    1) numbers 중 하나라도 doc 본문/메타에 있으면 HIT
       - 금액은 콤마 유무 차이가 있어서 digits_only 비교도 같이 한다
    2) phrases 중 하나라도 정규화된 doc 텍스트에 포함되면 HIT
       - 문장 완전일치까지는 아니고 "포함" 기준
    """
    meta = doc.metadata or {}
    meta_text = " ".join(str(v) for v in meta.values())
    raw_text = f"{doc.page_content or ''} {meta_text}"

    norm_text = normalize_text(raw_text)
    digit_text = digits_only(raw_text)

    # 숫자 기반 HIT
    for n in gold.get("numbers", []):
        if n and len(n) >= 6 and n in digit_text:
            return True, f"number:{n}"

    # 문장/구절 기반 HIT
    for p in gold.get("phrases", []):
        if p and p in norm_text:
            return True, f"phrase:{p[:50]}..."

    return False, ""


def eval_retriever_by_contexts(
    retriever,
    golden_path: str,
    k: int,
    debug: bool = True,
    debug_limit: int = 5,
) -> Dict[str, Any]:
    data = load_golden_json(golden_path)

    # 평가 가능한 샘플만 사용(질문 있고, gold 신호가 있어야 평가 가능)
    eval_items: List[Tuple[str, Dict[str, List[str]], Any]] = []
    for ex in data:
        q = (ex.get("question") or "").strip()
        gold = build_gold_signals(ex)
        if not q:
            continue
        if not gold["phrases"] and not gold["numbers"]:
            continue
        eval_items.append((q, gold, ex.get("id")))

    hits = 0
    mrr_sum = 0.0

    for idx, (query, gold, ex_id) in enumerate(eval_items, start=1):
        # retriever 자체가 k를 갖고 있어도 방어적으로 [:k]
        results = retriever.invoke(query)[:k]

        rank = None
        for i, d in enumerate(results, start=1):
            ok, _ = doc_contains_gold(d, gold)
            if ok:
                rank = i
                break
        # 임베딩
        # hit_reason = ""
        # hit_doc = None
        #
        # for i, d in enumerate(results, start=1):
        #     ok, reason = doc_contains_gold(d, gold)
        #     if ok:
        #         rank = i
        #         hit_reason = reason
        #         hit_doc = d
        #         break

        if rank is not None:
            hits += 1
            mrr_sum += 1.0 / rank

        # 디버그 로그
        if debug and idx <= debug_limit:
            print("\n" + "=" * 60)
            print(f"[DEBUG] #{idx} id={ex_id}")
            print("[Q]", query)
            print(f"[Result] rank={rank} (None이면 MISS)")

    n = len(eval_items)
    recall = hits / n if n else 0.0
    mrr = mrr_sum / n if n else 0.0

    return {"eval_n": n, f"recall@{k}": recall, f"mrr@{k}": mrr}

def load_team_json(json_path: str, add_type_prefix: bool = False) -> List[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for rec in data:
        system_id = rec.get("system_id") or ""
        content = rec.get("content", "") or ""
        metadata = rec.get("metadata", {}) or {}

        if not content.strip():
            continue

        metadata["system_id"] = system_id
        if add_type_prefix:
            t = metadata.get("type")
            if t:
                content = f"[{t}] {content}"

        docs.append(Document(page_content=content, metadata=metadata))

    return docs


#         if debug and idx <= debug_limit:
#             print("\n" + "=" * 60)
#             print(f"[DEBUG] #{idx}  id={ex_id}")
#             print("[Q]", query)
#
#             print("\n[Gold numbers] (상위 10개만)")
#             print(gold["numbers"][:10])
#
#             print("\n[Gold phrases] (상위 3개만)")
#             for p in gold["phrases"][:3]:
#                 print("-", p[:120])
#
#             if rank is None:
#                 print(f"\n[Result] MISS (Top{k} 안에서 hit 못 찾음)")
#                 if results:
#                     top1 = results[0]
#                     meta = top1.metadata or {}
#                     print("[Top1 source]", meta.get("source"), "chunk", meta.get("chunk_index"))
#                     print("[Top1 preview]", (top1.page_content or "")[:160].replace("\n", " "))
#             else:
#                 meta = (hit_doc.metadata or {}) if hit_doc else {}
#                 print(f"\n[Result] HIT at rank={rank}  reason={hit_reason}")
#                 print("[Hit source]", meta.get("source"), "chunk", meta.get("chunk_index"))
#                 print("[Hit preview]", (hit_doc.page_content or "")[:200].replace("\n", " "))
#
#     n = len(eval_items)
#     recall = hits / n if n else 0.0
#     mrr = mrr_sum / n if n else 0.0
#
#     print("\n===== Dense Retrieval Eval (contexts-based) =====")
#     print("golden_path:", resolve_path(golden_path))
#     print("eval_n     :", n)
#     print(f"Recall@{k} :", round(recall, 4))
#     print(f"MRR@{k}    :", round(mrr, 4))
#
#     return {
#         "eval_n": n,
#         f"recall@{k}": recall,
#         f"mrr@{k}": mrr,
#     }
#
#
# def load_team_json(json_path: str, add_type_prefix: bool = False) -> List[Document]:
#     """
#     전처리 JSON(list)을 LangChain Document 리스트로 변환
#     JSON record 예상 구조:
#     {
#       "system_id": "...",
#       "content": "...",   # 1000자 단위 chunk
#       "metadata": {...}   # source, type, chunk_index 등
#     }
#     """
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#
#     docs: List[Document] = []
#     for rec in data:
#         system_id = rec.get("system_id") or ""
#         content = rec.get("content", "") or ""
#         metadata = rec.get("metadata", {}) or {}
#
#         if not content.strip():
#             continue
#
#         metadata["system_id"] = system_id
#
#         if add_type_prefix:
#             t = metadata.get("type")
#             if t:
#                 content = f"[{t}] {content}"
#
#         docs.append(Document(page_content=content, metadata=metadata))
#
#     return docs

# def quick_sanity_check(retriever, query: str) -> None:
#     """
#     모델/인덱스가 제대로 매칭 됐는지 확인하는 디버그용
#     임베딩 모델이 최종으로 정해지고 파이프라인이 안정화되면 지워도 되지만
#     나중에 문서추가/전처리 변경/청크 규칙 변경되면 확인 가능 (안쓰면 주석 처리)
#     """
#     results = retriever.invoke(query)
#
#     print(f"\n[Query] {query}")
#     for i, doc in enumerate(results, start=1):
#         meta = doc.metadata
#         preview = doc.page_content[:200].replace("\n"," ")
#         print(f"\n--- Top {i} ---")
#         print("system_id    :", meta.get("system_id"))
#         print("type         :", meta.get("type"))
#         print("source       :", meta.get("source"))
#         print("chunk_index  :", meta.get("chunk_index"))
#         print("content_prev :", preview)
def quick_sanity_check(retriever, query: str) -> None:
    results = retriever.invoke(query)
    print(f"\n[Query] {query}")
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        preview = (doc.page_content or "")[:200].replace("\n", " ")
        print(f"\n--- Top {i} ---")
        print("source      :", meta.get("source"))
        print("chunk_index :", meta.get("chunk_index"))
        print("type        :", meta.get("type"))
        print("preview     :", preview)

# 조합 비교 + (선택) LangSmith Feedback 기록
# =======================================================
def run_compare_suite(vectorstore, docs: List[Document]) -> List[Dict[str, Any]]:
    """
    retriever_key × reranker_key 조합을 돌려서 결과를 리스트로 반환
    """
    # 너가 retriever_setting.py에 정의한 키를 그대로 쓰거나, 여기서 명시적으로 지정
    retriever_keys = ["dense", "bm25", "hybrid_rrf"]
    reranker_keys = ["none", "bge_rerank_m3", "bge_rerank_m3_ko"]

    results_all: List[Dict[str, Any]] = []

    for rk in retriever_keys:
        for rrk in reranker_keys:
            # retriever 생성(설정파일 안 바꾸고도 비교 가능)
            retriever = build_local_retriever(
                vectorstore=vectorstore,
                docs=docs,
                retriever_key=rk,
                reranker_key=rrk,
            )

            # invoke 반복이 trace에 너무 많이 쌓이면 끄기
            with ls.tracing_context(enabled=TRACE_EVAL_CALLS):
                metrics = eval_retriever_by_contexts(
                    retriever=retriever,
                    golden_path=GOLDEN_JSON_PATH,
                    k=int(rs.TOP_K),       # TOP_K는 retriever_setting 기준으로 고정
                    debug=False,
                )

            row = {
                "retriever": rk,
                "reranker": rrk,
                **metrics,
            }
            results_all.append(row)

            print(
                f"[DONE] {rk} + {rrk} | "
                f"Recall@{rs.TOP_K}={metrics.get(f'recall@{rs.TOP_K}', 0):.4f} | "
                f"MRR@{rs.TOP_K}={metrics.get(f'mrr@{rs.TOP_K}', 0):.4f} | "
                f"eval_n={metrics.get('eval_n')}"
            )

            # ✅ LangSmith feedback 기록(조합별 key로 저장)
            if ENABLE_LANGSMITH and LOG_FEEDBACK:
                try:
                    rt = get_current_run_tree()
                    trace_id = getattr(rt, "trace_id", rt.id)
                    client = Client()

                    recall_key = f"{rk}__{rrk}__recall@{rs.TOP_K}"
                    mrr_key = f"{rk}__{rrk}__mrr@{rs.TOP_K}"

                    if f"recall@{rs.TOP_K}" in metrics:
                        client.create_feedback(
                            trace_id=trace_id,
                            key=recall_key,
                            score=float(metrics[f"recall@{rs.TOP_K}"]),
                        )
                    if f"mrr@{rs.TOP_K}" in metrics:
                        client.create_feedback(
                            trace_id=trace_id,
                            key=mrr_key,
                            score=float(metrics[f"mrr@{rs.TOP_K}"]),
                        )
                except Exception as e:
                    print("[WARN] Feedback 기록 실패:", e)

    return results_all

@traceable(name="build_index_run_dense_eval")
def main() -> Dict[str, Any]:
    # 데이터 로딩
    docs = load_team_json(JSON_PATH, add_type_prefix=ADD_TYPE_PREFIX)
    print(f"Loaded docs: {len(docs)}")

    # 샘플링(임베딩 모델 비교 단계)
    if MAX_DOCS and len(docs) > MAX_DOCS:
        docs = docs[:MAX_DOCS]
        print(f"Sampled docs: {len(docs)} (MAX_DOCS={MAX_DOCS})")
    spec = get_spec(EMBEDDING_KEY)
    print(f"Embedding key : {EMBEDDING_KEY}")
    print(f"Provider      : {spec.provider}")
    print(f"Model name    : {spec.model_name}")

    embeddings = get_embeddings(
        EMBEDDING_KEY,
        normalize=NORMALIZE_EMBEDDINGS,
        device=DEVICE,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # ✅ 여기서 조합 비교 실행
    suite = run_compare_suite(vectorstore=vectorstore, docs=docs)

    # (선택) 사람이 보기 쉽게 top 몇 개만 뽑아서 요약 반환
    # 여기서는 MRR 기준 정렬
    best = sorted(suite, key=lambda x: x.get(f"mrr@{rs.TOP_K}", 0), reverse=True)[:5]

    # (선택) sanity check는 “최고 조합 1개”로만 해도 됨
    if best:
        rk = best[0]["retriever"]
        rrk = best[0]["reranker"]
        print(f"\n[Sanity] best = {rk} + {rrk}")
        retriever_best = build_local_retriever(
            vectorstore=vectorstore,
            docs=docs,
            retriever_key=rk,
            reranker_key=rrk,
        )
        with ls.tracing_context(enabled=TRACE_SANITY):
            quick_sanity_check(retriever_best, query="대학 관련 사업의 공고번호와 사업명은 무엇인가요?")

    return {"top5_by_mrr": best, "all_results": suite}


if __name__ == "__main__":
    with ls.tracing_context(enabled=ENABLE_LANGSMITH):
        out = main()
        print("\n[MAIN RETURN]", out)

# 임베딩
# def quick_sanity_check(retriever, query: str) -> None:
#     results = retriever.invoke(query)
#     print(f"\n[Query] {query}")
#     for i, doc in enumerate(results, start=1):
#         meta = doc.metadata or {}
#         preview = (doc.page_content or "")[:200].replace("\n", " ")
#         print(f"\n--- Top {i} ---")
#         print("source      :", meta.get("source"))
#         print("chunk_index :", meta.get("chunk_index"))
#         print("type        :", meta.get("type"))
#         print("preview     :", preview)
#
#
#     # 어떤 모델이 선택됐는지 출력(로그용)
#     spec = get_spec(EMBEDDING_KEY)
#     print(f"Embedding key  : {EMBEDDING_KEY}")
#     print(f"Provider       : {spec.provider}")
#     print(f"Model name     : {spec.model_name}")
#
#     # 임베딩 객체 생성
#     embeddings = get_embeddings(
#         EMBEDDING_KEY,
#         normalize=NORMALIZE_EMBEDDINGS,
#         device=DEVICE,
#     )
#     # 벡터스토어 생성
#     vectorstore = FAISS.from_documents(docs, embeddings)
#
#     # Retriever
#     retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
#
#     # -------------------------------
#     # 평가(Recall/MRR)
#     # - TRACE_EVAL_CALLS=False면 retriever.invoke 반복 호출이 Trace에 쌓이지 않음(깔끔)
#     # -------------------------------
#     with ls.tracing_context(enabled=TRACE_EVAL_CALLS):
#         metrics = eval_retriever_by_contexts(
#             retriever=retriever,
#             golden_path=GOLDEN_JSON_PATH,
#             k=TOP_K,
#             debug=True,
#         )
#
#         # -------------------------------
#         # (선택) LangSmith Feedback로도 기록
#         # -------------------------------
#         if ENABLE_LANGSMITH and LOG_FEEDBACK:
#             try:
#                 rt = get_current_run_tree()
#                 trace_id = getattr(rt, "trace_id", rt.id)  # SDK 버전에 따라 trace_id가 없을 수 있어 안전 처리
#                 client = Client()
#
#                 recall_key = f"recall@{TOP_K}"
#                 mrr_key = f"mrr@{TOP_K}"
#
#                 if recall_key in metrics:
#                     client.create_feedback(
#                         trace_id=trace_id,
#                         key=recall_key,
#                         score=float(metrics[recall_key]),
#                     )
#                 if mrr_key in metrics:
#                     client.create_feedback(
#                         trace_id=trace_id,
#                         key=mrr_key,
#                         score=float(metrics[mrr_key]),
#                     )
#             except Exception as e:
#                 print("[WARN] Feedback 기록 실패:", e)
#         # -------------------------------
#         # 간단 retrieval 테스트(원하면 토글로 trace 끄기)
#         # -------------------------------
#         queries = [
#             "대학 관련 사업의 공고번호와 사업명은 무엇인가요?",
#             "차세대 관련 사업의 사업명과 사업요약은 무엇인가요?",
#             "한영대학교 제안요청서에서 사업명과 공고번호는 어떻게 되나요?",
#         ]
#
#         with ls.tracing_context(enabled=TRACE_SANITY):
#             for q in queries:
#                 quick_sanity_check(retriever, query=q)
#
#         return metrics
#
#     # 간단 retrieval 테스트
#     # quick_sanity_check(retriever, query="입찰 참여 마감일은 언제야?")
#
# if __name__ == "__main__":
#     with ls.tracing_context(enabled=ENABLE_LANGSMITH):
#         out = main()
#         print("\n[MAIN RETURN]", out)
