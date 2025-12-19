# processing/chunking_json.py
import os
import json
import re
import uuid

BASE_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data\output"
INPUT_DIR = os.path.join(BASE_DIR, "json_processed")
OUT_DIR = os.path.join(BASE_DIR, "json_chunking")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------
# heading 패턴 (라인 내부 탐지용)
# ---------------------------------------
SECTION_PATTERNS = [
    (1, re.compile(r"(Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ)\.?\s*[가-힣A-Za-z]+")),
    (2, re.compile(r"\b\d+\.\s*[가-힣A-Za-z]+")),
    (3, re.compile(r"\b\d+\)\s*[가-힣A-Za-z]+")),
    (3, re.compile(r"[가-하]\.\s*[가-힣A-Za-z]+")),
]


def extract_headings_from_line(line):
    """
    한 줄에서 section/subsection 후보를 순서대로 추출
    """
    found = []
    for level, pattern in SECTION_PATTERNS:
        match = pattern.search(line)
        if match:
            title = match.group().strip()
            if len(title) <= 60:
                found.append((level, title))
    return found


# ---------------------------------------
# streaming chunk 생성 (메모리 안전)
# ---------------------------------------
def stream_chunks(lines, max_size=900, overlap=150):
    buffer = ""
    overlap_buf = ""

    for line in lines:
        if len(buffer) + len(line) > max_size:
            if buffer.strip():
                yield buffer.strip()
            overlap_buf = buffer[-overlap:] if overlap < len(buffer) else buffer
            buffer = overlap_buf + "\n" + line
        else:
            buffer = buffer + "\n" + line if buffer else line

    if buffer.strip():
        yield buffer.strip()


# ---------------------------------------
# JSON 1개 파일 처리
# ---------------------------------------
def process_processed_json(path, idx, total, max_size=900, overlap=150):
    file_name = os.path.basename(path)
    print(f"\n[{idx}/{total}] 처리 시작: {file_name}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    current_section = None
    current_subsection = None
    final_chunks = []

    pages = data.get("pages", [])
    total_pages = len(pages)

    for p_idx, page in enumerate(pages, start=1):
        page_number = page.get("page_number")
        text = page.get("clean_text", "")
        tables = page.get("clean_tables", [])

        print(f"  - 페이지 {p_idx}/{total_pages} (page={page_number}) 처리중")

        if not text.strip():
            continue

        lines_for_chunk = []

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # 🔑 핵심: 라인 내부에서 제목 추출
            headings = extract_headings_from_line(line)
            for level, title in headings:
                if level == 1:
                    current_section = title
                    current_subsection = None
                else:
                    current_subsection = title

            lines_for_chunk.append(line)

        # 테이블은 뒤에 붙임
        for table in tables:
            if table:
                lines_for_chunk.append(json.dumps(table, ensure_ascii=False))

        if not lines_for_chunk:
            continue

        chunks = list(stream_chunks(lines_for_chunk, max_size, overlap))

        for ch in chunks:
            if not ch.strip():
                continue

            final_chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk": ch,
                    "page": page_number,
                    "section": current_section,
                    "subsection": current_subsection,
                    "source_file": file_name,
                    "length": len(ch),
                }
            )

    out_path = os.path.join(
        OUT_DIR, file_name.replace("_processed.json", "_chunking.json")
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=4)

    print(f"✔ 저장 완료 → {out_path}")


# ---------------------------------------
# 실행부
# ---------------------------------------
if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_processed.json")]
    total = len(files)

    for i, f in enumerate(files, start=1):
        process_processed_json(os.path.join(INPUT_DIR, f), i, total)
