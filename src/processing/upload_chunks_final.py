#==================================================================
# 프로그램명: upload_chunks_with_prev_context.py
# 설명:
# - 새로 생성한 chunks_all_pdfs_final.json 로드
# - final_classification_hierarchy.csv에서 summary/category/depth 등 메타를 파일명 기준으로 매칭
# - prev_context n_token을 같은 source_file 내 chunk_index 순서로 누적하여 embedding 입력에 포함
# - documents_chunks_smk_2에 upsert(충돌 방지: chunk_id 기준)
# - 파일명 매칭: trim + NBSP/BOM 제거 + basename + stem(strip) 매칭 + 보조키(공백/구분자 정규화) 매칭
# - 텍스트 내 \u0000 같은 NUL 제거(Postgres text/JSON/embedding 입력 안전)
# - timestamptz 입력 안전: "YYYY-MM-DD HH:MM:SS" -> ISO8601로 정규화
# - embedding_input: 사업명 + 메타데이터(지정 항목) + (첫 청크만) 사업요약(CSV) + prev_context + 텍스트
# - metadata.embedding_source: "project_name + metadata + (summary only first chunk) + prev_context + text"
#==================================================================

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re
import pandas as pd
import tiktoken
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

TABLE_NAME = "documents_chunks_smk_3"
CSV_PATH = BASE_DIR / "data" / "final_classification_hierarchy.csv"

CHUNKS_JSON_PATH = BASE_DIR / "data" / "chunks_all_pdfs_final.json"

PREV_CONTEXT_TOKENS = 300
SUMMARY_MAX_CHARS = 1200

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

enc = tiktoken.get_encoding("cl100k_base")

# -----------------------------
# helpers (string / filename)
# -----------------------------
def clean_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u00a0", " ")  # NBSP
    return s

def strip_nul(text: str) -> str:
    return (text or "").replace("\x00", "").replace("\u0000", "")

def norm_basename(s: str) -> str:
    s = clean_str(s).strip()
    s = os.path.basename(s)
    return s.lower()

def stem_of(filename: str) -> str:
    b = norm_basename(filename)
    st = Path(b).stem
    return st.strip().lower()

def loose_key(s: str) -> str:
    s = clean_str(s).lower()
    s = os.path.basename(s)
    s = Path(s).stem
    s = s.strip()
    s = re.sub(r"[\s\-_\.]+", "", s)
    return s

def to_text_or_none(v):
    if v is None:
        return None
    if pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None

def to_int_or_none(v):
    if v is None or pd.isna(v):
        return None
    try:
        return int(float(v))
    except Exception:
        return None

def to_bigint_or_none(v):
    if v is None or pd.isna(v):
        return None
    try:
        return int(float(v))
    except Exception:
        return None

def to_iso_timestamptz_or_none(v):
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.isoformat()
    except Exception:
        return None

# -----------------------------
# embedding
# -----------------------------
def embed_text(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def last_n_tokens(text: str, n_tokens: int) -> str:
    text = strip_nul(text or "")
    if not text:
        return ""
    ids = enc.encode(text)
    if len(ids) <= n_tokens:
        return text
    return enc.decode(ids[-n_tokens:])

# -----------------------------
# CSV meta map (stem key + loose key)
# -----------------------------
def build_csv_meta_map(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    stem_map = {}
    loose_map = {}

    for fname, g in df.groupby("파일명", dropna=False):
        if pd.isna(fname):
            continue

        row = g.iloc[0]
        meta = {
            "summary": to_text_or_none(row.get("사업 요약")),
            "category": to_text_or_none(row.get("Category")),
            "category_llm": to_text_or_none(row.get("Category_LLM")),
            "depth_1": to_text_or_none(row.get("Depth_1")),
            "depth_2": to_text_or_none(row.get("Depth_2")),
        }

        st = stem_of(fname)
        lk = loose_key(fname)

        if st:
            stem_map[st] = meta
        if lk:
            loose_map[lk] = meta

    return stem_map, loose_map

def find_csv_meta(source_file: str, stem_map: dict, loose_map: dict):
    st = stem_of(source_file)
    if st and st in stem_map:
        return stem_map[st]

    lk = loose_key(source_file)
    if lk and lk in loose_map:
        return loose_map[lk]

    return None

# -----------------------------
# build embedding input (project_name + metadata + (summary only first chunk) + prev_context + text)
# - 메타데이터에는 depth/category/category_llm/summary를 넣지 않음
# - 사업요약(summary)은 CSV에서만 가져오며, 각 파일의 첫 청크에만 포함
# -----------------------------
def build_embedding_input(chunk: dict, csv_meta: dict | None, prev_context: str, include_summary: bool) -> str:
    # chunk 값(=요청한 항목들 중 chunk에서 오는 것)
    공고_번호 = strip_nul(str(chunk.get("announcement_id") or ""))
    공고_차수 = strip_nul(str(chunk.get("announcement_round") or ""))
    사업명 = strip_nul(chunk.get("project_name") or "")
    사업_금액 = strip_nul(str(chunk.get("project_budget") or ""))
    발주_기관 = strip_nul(chunk.get("ordering_agency") or "")
    공개_일자 = strip_nul(str(chunk.get("published_at") or ""))
    입찰_참여_시작일 = strip_nul(str(chunk.get("bid_start_at") or ""))
    입찰_참여_마감일 = strip_nul(str(chunk.get("bid_end_at") or ""))
    파일형식 = strip_nul(chunk.get("file_type") or "")
    파일명 = strip_nul(chunk.get("source_file") or "")
    텍스트 = strip_nul(chunk.get("text") or "")

    # CSV에서 가져오는 항목들(요약만, 첫 청크에만)
    사업_요약 = ""
    if include_summary:
        사업_요약 = strip_nul((csv_meta or {}).get("summary") or "")
        if len(사업_요약) > SUMMARY_MAX_CHARS:
            사업_요약 = 사업_요약[:SUMMARY_MAX_CHARS]

    # ✅ metadata 블록(요약/카테고리/뎁스 제외)
    meta_lines = []
    if 공고_번호: meta_lines.append(f"공고 번호: {공고_번호}")
    if 공고_차수: meta_lines.append(f"공고 차수: {공고_차수}")
    if 사업명: meta_lines.append(f"사업명: {사업명}")
    if 사업_금액: meta_lines.append(f"사업 금액: {사업_금액}")
    if 발주_기관: meta_lines.append(f"발주 기관: {발주_기관}")
    if 공개_일자: meta_lines.append(f"공개 일자: {공개_일자}")
    if 입찰_참여_시작일: meta_lines.append(f"입찰 참여 시작일: {입찰_참여_시작일}")
    if 입찰_참여_마감일: meta_lines.append(f"입찰 참여 마감일: {입찰_참여_마감일}")
    if 파일형식: meta_lines.append(f"파일형식: {파일형식}")
    if 파일명: meta_lines.append(f"파일명: {파일명}")

    parts = []
    if 사업명:
        parts.append(f"[사업명]\n{사업명}")

    parts.append("[메타데이터]")
    parts.append("\n".join(meta_lines).strip())

    if include_summary:
        parts.append("[사업 요약]")
        parts.append(사업_요약)

    if prev_context:
        parts.append("[이전문맥]")
        parts.append(strip_nul(prev_context))

    parts.append("[텍스트]")
    parts.append(텍스트)

    return "\n".join([p for p in parts if p]).strip()

# -----------------------------
# build row for DB insert (documents_chunks_smk_2)
# - metadata에서 depth_1/depth_2/summary 제외
# -----------------------------
def build_db_row(chunk: dict, csv_meta: dict | None, embedding: list[float]):
    md = chunk.get("metadata") or {}
    content_type = md.get("content_type")
    chunk_index = md.get("chunk_index")

    category = to_text_or_none((csv_meta or {}).get("category"))
    category_llm = to_text_or_none((csv_meta or {}).get("category_llm"))
    depth_1 = to_text_or_none((csv_meta or {}).get("depth_1"))
    depth_2 = to_text_or_none((csv_meta or {}).get("depth_2"))

    path = []
    if depth_1: path.append(depth_1)
    if depth_2: path.append(depth_2)
    path = path or None

    out_metadata = {
        "lang": "ko",
        "path": path,
        "type": content_type,
        "category": category,
        "category_llm": category_llm,
        "embedding_source": "project_name + metadata + (summary only first chunk) + prev_context + text",
    }

    return {
        "chunk_id": chunk["chunk_id"],
        "pages": chunk["pages"],

        "announcement_id": chunk.get("announcement_id"),
        "announcement_round": to_int_or_none(chunk.get("announcement_round")),
        "project_name": chunk.get("project_name"),
        "project_budget": to_bigint_or_none(chunk.get("project_budget")),
        "ordering_agency": chunk.get("ordering_agency"),

        "published_at": to_iso_timestamptz_or_none(chunk.get("published_at")),
        "bid_start_at": to_iso_timestamptz_or_none(chunk.get("bid_start_at")),
        "bid_end_at": to_iso_timestamptz_or_none(chunk.get("bid_end_at")),

        "text": strip_nul(chunk.get("text") or ""),
        "length": to_int_or_none(chunk.get("length")) or len(strip_nul(chunk.get("text") or "")),

        "content_type": content_type,
        "chunk_index": to_int_or_none(chunk_index),

        "source_file": chunk.get("source_file"),
        "file_type": chunk.get("file_type"),

        "metadata": out_metadata,
        "embedding": embedding,
    }

# -----------------------------
# main
# -----------------------------
def main():
    stem_map, loose_map = build_csv_meta_map(CSV_PATH)
    print("CSV meta(stem) count:", len(stem_map))
    print("CSV meta(loose) count:", len(loose_map))

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("Chunks loaded:", len(chunks))

    # prev_context는 파일 단위로 유지해야 하므로 (source_file, chunk_index) 기준 정렬
    def key_fn(c):
        sf = c.get("source_file") or ""
        ci = (c.get("metadata") or {}).get("chunk_index")
        try:
            ci = int(ci)
        except Exception:
            ci = 0
        return (norm_basename(sf), ci)

    chunks = sorted(chunks, key=key_fn)

    no_match_files = set()
    inserted = 0
    failed = 0
    failed_rows = []  # ✅ 실패 chunk_id/파일명 기록

    prev_file = None
    prev_tail = ""

    for chunk in tqdm(chunks, desc="Uploading chunks", unit="chunk"):
        source_file = chunk.get("source_file") or ""
        cur_file = norm_basename(source_file)

        # 파일 바뀌면 prev_context 리셋
        is_first_chunk_of_file = (prev_file != cur_file)
        if is_first_chunk_of_file:
            prev_tail = ""
            prev_file = cur_file

        csv_meta = find_csv_meta(source_file, stem_map, loose_map)
        if not csv_meta:
            no_match_files.add(source_file)

        prev_context = prev_tail

        emb_input = build_embedding_input(
            chunk=chunk,
            csv_meta=csv_meta,
            prev_context=prev_context,
            include_summary=is_first_chunk_of_file
        )

        try:
            emb = embed_text(emb_input)
            row = build_db_row(chunk, csv_meta, emb)
            supabase.table(TABLE_NAME).upsert(row, on_conflict="chunk_id").execute()
            inserted += 1
        except Exception as e:
            failed += 1
            failed_rows.append((chunk.get("chunk_id"), source_file, str(e)))
            print(f"\n❌ upload fail chunk_id={chunk.get('chunk_id')} file={source_file}\n   {e}\n")

        # 다음 chunk를 위한 prev_tail 갱신
        prev_tail = last_n_tokens(chunk.get("text") or "", PREV_CONTEXT_TOKENS)

    # CSV 매칭 실패 파일 저장
    out_path = BASE_DIR / "data" / "embedding_no_match_files_smk_2.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for fn in sorted(no_match_files):
            f.write(fn + "\n")

    # 업로드 실패 chunk 저장
    fail_path = BASE_DIR / "data" / "upload_failed_chunks_smk_2.txt"
    with open(fail_path, "w", encoding="utf-8") as f:
        for cid, fn, err in failed_rows:
            f.write(f"{cid}\t{fn}\t{err}\n")

    print("\n=== DONE ===")
    print("inserted/upserted:", inserted)
    print("failed:", failed)
    print("no_match_files:", len(no_match_files))
    print("saved(no_match_files):", out_path)
    print("saved(failed_chunks):", fail_path)

if __name__ == "__main__":
    main()
