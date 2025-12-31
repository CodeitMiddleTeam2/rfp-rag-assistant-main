#==================================================================
#프로그램명: upload_chunks_final.py
#설명:

#- final_classification_hierarchy.csv를 로드하여 파일명 stem 기준으로 PDF 메타데이터 매핑 생성
#- converted_pdfs 폴더 내 모든 PDF 순회 처리
#- pdfplumber + PyMuPDF로 표 / 본문을 분리 추출하며, PyMuPDF 실패 시 pdfplumber로 폴백
#- 페이지 내 표 영역(bbox)을 제외한 본문만 클리핑 추출
#- 페이지 전반에 반복 등장하는 헤더/푸터를 상단 TOP_N, 하단 BOTTOM_N, 출현 비율 MIN_RATIO 기준으로 제거
#- 유니코드 NFC 정규화 및 컨트롤 문자(NUL 포함) 제거, 잡음 라인(구분선, 페이지 번호 등) 필터링
#- 연속 반복되는 동일 라인(run)만 1개로 축약하여 중복 텍스트 제거
#- tiktoken(cl100k_base) 기준 토큰 청킹
#  · 본문: TEXT_MAX_TOKENS, 라인 오버랩(TEXT_OVERLAP_LINES) 적용
#  · 표: TABLE_MAX_TOKENS, 행 단위 청킹 (오버랩 없음)
#- 각 chunk에 UUID 기반 chunk_id 부여 및 페이지 매핑 유지
#- CSV 메타데이터 + pages + text + length + metadata(content_type, chunk_index) 구조로 레코드 생성
#- 결과를 chunks_all_pdfs_final.json 파일로 저장
#- 처리 완료 후 총 chunk 수, CSV 매칭 실패로 스킵된 PDF 수, run 축약으로 제거된 라인 수 출력
#- 파일명 매칭 보정 로직
#  · basename/stem 기반 매칭
#  · BOM, NBSP 제거 및 대소문자 통일
#  · "abc .pdf" 형태처럼 확장자 앞 공백이 있는 경우도 정규화하여 CSV ↔ PDF 매칭 실패 방지
#
#==================================================================


import os
import json
import uuid
import re
import unicodedata
from collections import Counter

import pandas as pd
import pdfplumber
import tiktoken
import fitz  # PyMuPDF


# =============================
# 설정
# =============================
CSV_PATH  = r"C:\Users\USER\Desktop\project_team2\data\final_classification_hierarchy.csv"
FILES_DIR = r"C:\Users\USER\Desktop\project_team2\data\converted_pdfs"
OUT_PATH  = r"C:\Users\USER\Desktop\project_team2\data\chunks_all_pdfs_final.json"

TEXT_MAX_TOKENS = 1024
TEXT_OVERLAP_LINES = 3
TABLE_MAX_TOKENS = 1200

TOP_N = 3
BOTTOM_N = 3
MIN_RATIO = 0.6

enc = tiktoken.get_encoding("cl100k_base")


# =============================
# 유니코드/컨트롤 문자 방어 (추가)
# =============================
def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return s


def strip_controls(s: str) -> str:
    # 컨트롤 문자 제거 (예: 2024.\x01 10.)
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


# =============================
# ✅ 연속 반복(run)만 1개 남김 (추가)
# =============================
def collapse_consecutive_duplicates(lines):
    """
    - 같은 라인이 연속으로 반복될 때만 1개로 축약
    - 비연속 반복은 건드리지 않음
    """
    out = []
    prev = None
    removed = 0
    for l in lines:
        if l == prev:
            removed += 1
            continue
        out.append(l)
        prev = l
    return out, removed


# =============================
# 1) PDF에서 표/본문 분리 추출
# =============================
def extract_table_lines(pl_page):
    out = []
    tables = pl_page.extract_tables()
    if not tables:
        return out

    for tbl in tables:
        for row in tbl:
            cells = []
            for cell in row:
                if cell is None:
                    continue
                cell = " ".join(safe_str(cell).splitlines()).strip()
                cell = strip_controls(cell)
                if cell:
                    cells.append(cell)
            if cells:
                out.append(" | ".join(cells))
    return out


def extract_table_bboxes(pl_page):
    return [t.bbox for t in pl_page.find_tables()]


def safe_page_text_lines(fz_doc, page_index, pl_page, clip=None):
    """
    ✅ PyMuPDF 터지면(pdf 구조 깨짐 등) pdfplumber.extract_text로 폴백
    """
    # 1) fitz 우선
    if fz_doc is not None:
        try:
            p = fz_doc[page_index]
            txt = p.get_text("text", clip=clip) if clip else p.get_text("text")
            lines = []
            for l in txt.splitlines():
                l = strip_controls(safe_str(l).strip())
                if l:
                    lines.append(l)
            return lines
        except Exception:
            pass

    # 2) pdfplumber fallback
    try:
        txt2 = pl_page.extract_text() or ""
        lines = []
        for l in txt2.splitlines():
            l = strip_controls(safe_str(l).strip())
            if l:
                lines.append(l)
        return lines
    except Exception:
        return []


def compute_non_table_strips(page_bbox, table_bboxes, min_height=20):
    x0, y0, x1, y1 = page_bbox
    if not table_bboxes:
        return [(x0, y0, x1, y1)]

    intervals = sorted([(b[1], b[3]) for b in table_bboxes])
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    strips = []
    cur = y0
    for a, b in merged:
        if a - cur >= min_height:
            strips.append((x0, cur, x1, a))
        cur = max(cur, b)

    if y1 - cur >= min_height:
        strips.append((x0, cur, x1, y1))

    return strips


def extract_normal_lines_excluding_tables(fz_doc, page_index, pl_page):
    table_bboxes = extract_table_bboxes(pl_page)
    strips = compute_non_table_strips(pl_page.bbox, table_bboxes)

    lines = []
    for (x0, y0, x1, y1) in strips:
        clip = fitz.Rect(x0, y0, x1, y1)
        lines.extend(safe_page_text_lines(fz_doc, page_index, pl_page, clip=clip))
    return lines


def extract_pdf_lines_separately(pdf_path):
    table_pages = []
    text_pages = []

    with pdfplumber.open(pdf_path) as pl_pdf:
        try:
            fz_doc = fitz.open(pdf_path)
        except Exception:
            fz_doc = None

        for i, pl_page in enumerate(pl_pdf.pages):
            table_pages.append(extract_table_lines(pl_page))
            text_pages.append(extract_normal_lines_excluding_tables(fz_doc, i, pl_page))

        if fz_doc is not None:
            fz_doc.close()

    return table_pages, text_pages


# =============================
# 2) 헤더/푸터 제거
# =============================
def remove_repeating_header_footer(pages, top_n=3, bottom_n=3, min_ratio=0.6):
    total = len(pages)
    if total == 0:
        return pages

    top = Counter()
    bottom = Counter()

    for lines in pages:
        for l in lines[:top_n]:
            top[l] += 1
        for l in lines[-bottom_n:]:
            bottom[l] += 1

    header_lines = {l for l, c in top.items() if c / total >= min_ratio}
    footer_lines = {l for l, c in bottom.items() if c / total >= min_ratio}

    return [[l for l in lines if l not in header_lines and l not in footer_lines] for lines in pages]


# =============================
# 3) 라인 정제 (베이스 로직 유지 + 안전 처리만 적용)
# =============================
def clean_lines(lines):
    out = []
    for s in lines:
        s = strip_controls(safe_str(s).strip())
        if not s:
            continue
        s = s.replace("­", "")
        if re.fullmatch(r"[-=*_]{3,}", s):
            continue
        if re.fullmatch(r"-?\s*\d+\s*-?", s):
            continue
        if re.fullmatch(r"(?i)page\s*\d+", s):
            continue
        if s.count("·") >= 10:
            continue
        out.append(s)
    return out


# =============================
# 4) 청킹 + 페이지 매핑
# =============================
def chunk_lines_by_tokens_with_page_map(paged_lines, max_tokens, overlap_lines=3):
    chunks, buf, buf_tok = [], [], 0

    def tok_len(s):
        return len(enc.encode(s))

    def flush():
        if not buf:
            return
        chunks.append({
            "text": "\n".join(l for _, l in buf),
            "pages": sorted({p for p, _ in buf}),
        })

    for page_id, line in paged_lines:
        t = tok_len(line) + 1

        if t >= max_tokens:
            flush()
            buf.clear()
            chunks.append({"text": line, "pages": [page_id]})
            continue

        if buf_tok + t > max_tokens and buf:
            flush()
            buf[:] = buf[-overlap_lines:]
            buf_tok = tok_len("\n".join(l for _, l in buf)) if buf else 0

        buf.append((page_id, line))
        buf_tok += t

    flush()
    return chunks


def chunk_table_rows_with_page_map(paged_rows, max_tokens=600):
    chunks, buf, buf_tok = [], [], 0

    def flush():
        if not buf:
            return
        chunks.append({
            "text": "\n".join(r for _, r in buf),
            "pages": sorted({p for p, _ in buf}),
        })

    for page_id, r in paged_rows:
        rt = len(enc.encode(r)) + 1

        if rt >= max_tokens:
            flush()
            buf.clear()
            chunks.append({"text": r, "pages": [page_id]})
            continue

        if buf_tok + rt > max_tokens and buf:
            flush()
            buf.clear()
            buf_tok = 0

        buf.append((page_id, r))
        buf_tok += rt

    flush()
    return chunks


# =============================
# 5) 실행 (PDF 폴더 전부 처리 + CSV 메타 매칭)
# =============================
def norm_stem(name: str) -> str:
    """
    확장자(.hwp/.pdf) 제거한 stem으로 매칭
    ✅ CSV에 "abc .hwp" 처럼 확장자 앞 공백이 있어도 매칭되게 보정
    """
    if name is None:
        return ""
    s = safe_str(name)
    s = s.strip()
    s = os.path.basename(s)
    s = s.replace("\ufeff", "").replace("\u00a0", " ")
    s = s.lower()

    # ✅ "abc .hwp" / "abc .pdf" -> "abc.hwp" / "abc.pdf"
    s = re.sub(r"\s+\.(pdf|hwp)$", r".\1", s)

    if s.endswith(".pdf"):
        s = s[:-4]
    elif s.endswith(".hwp"):
        s = s[:-4]

    return s.strip().rstrip(".").strip()


df = pd.read_csv(CSV_PATH)

# CSV 메타를 stem 기준으로 맵 구성 (hwp/pdf 섞여 있어도 동일 stem이면 매칭됨)
meta_map = {}
for _, row in df.iterrows():
    stem = norm_stem(row.get("파일명"))
    if not stem:
        continue

    base_meta = {
        "source_file": row.get("파일명"),          
        "project_name": row.get("사업명"),
        "ordering_agency": row.get("발주 기관"),
        "project_budget": row.get("사업 금액"),
        "bid_start_at": row.get("입찰 참여 시작일"),
        "bid_end_at": row.get("입찰 참여 마감일"),
        "announcement_id": row.get("공고 번호"),
        "announcement_round": row.get("공고 차수"),
        "published_at": row.get("공개 일자"),
        "summary": row.get("사업 요약"),
        "file_type": row.get("파일형식"),
    }
    base_meta = {k: (None if pd.isna(v) else v) for k, v in base_meta.items()}
    meta_map[stem] = base_meta


records = []

# converted_pdfs 폴더 안 PDF 전부 처리
pdf_files = [f for f in os.listdir(FILES_DIR) if f.lower().endswith(".pdf")]
pdf_files.sort()

skipped = 0

# 연속 반복(run) 축약 통계
run_removed_text = 0
run_removed_table = 0

for idx, fn in enumerate(pdf_files, 1):
    # ✅ TQRM = 전체 진행률
    print(f"[TQRM] {idx}/{len(pdf_files)}: {fn}")

    pdf_path = os.path.join(FILES_DIR, fn)
    stem = norm_stem(fn)

    if stem not in meta_map:
        print(f"⚠️ CSV 메타 매칭 실패 -> 스킵: {fn}")
        skipped += 1
        continue

    base_meta = dict(meta_map[stem])

    # ✅ source_file은 "실제로 청킹한 변환 PDF 파일명"으로 저장
    base_meta["source_file"] = fn

    table_pages, text_pages = extract_pdf_lines_separately(pdf_path)

    table_pages = remove_repeating_header_footer(table_pages, top_n=TOP_N, bottom_n=BOTTOM_N, min_ratio=MIN_RATIO)
    text_pages  = remove_repeating_header_footer(text_pages,  top_n=TOP_N, bottom_n=BOTTOM_N, min_ratio=MIN_RATIO)

    table_pages = [clean_lines(p) for p in table_pages]
    text_pages  = [clean_lines(p) for p in text_pages]

    # ✅ "연속 반복(run)"만 1개 남김 (여기만 적용)
    new_text_pages = []
    for p in text_pages:
        p2, removed = collapse_consecutive_duplicates(p)
        run_removed_text += removed
        new_text_pages.append(p2)
    text_pages = new_text_pages

    new_table_pages = []
    for p in table_pages:
        p2, removed = collapse_consecutive_duplicates(p)
        run_removed_table += removed
        new_table_pages.append(p2)
    table_pages = new_table_pages

    # ---- text ----
    paged_text = [(pid, l) for pid, p in enumerate(text_pages, 1) for l in p]
    text_chunks = chunk_lines_by_tokens_with_page_map(paged_text, TEXT_MAX_TOKENS, TEXT_OVERLAP_LINES)

    for i, ch in enumerate(text_chunks):
        records.append({
            "chunk_id": str(uuid.uuid4()),
            "pages": ch["pages"],
            **base_meta,
            "text": ch["text"],
            "length": len(ch["text"]),
            "metadata": {
                "content_type": "text",
                "chunk_index": i,
            }
        })

    # ---- table ----
    paged_tables = [(pid, r) for pid, p in enumerate(table_pages, 1) for r in p]
    table_chunks = chunk_table_rows_with_page_map(paged_tables, TABLE_MAX_TOKENS)

    for i, ch in enumerate(table_chunks):
        records.append({
            "chunk_id": str(uuid.uuid4()),
            "pages": ch["pages"],
            **base_meta,
            "text": ch["text"],
            "length": len(ch["text"]),
            "metadata": {
                "content_type": "table",
                "chunk_index": i,
            }
        })

    print(f"✅ 처리 완료: {fn} (text={len(text_chunks)}, table={len(table_chunks)})")


with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("saved:", OUT_PATH)
print("total chunks:", len(records))
print("skipped pdfs (no csv meta):", skipped)
print("run_removed_text_lines:", run_removed_text)
print("run_removed_table_lines:", run_removed_table)
