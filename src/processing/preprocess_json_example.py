import os
import json
import re

BASE_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data\output"
RAW_DIR = os.path.join(BASE_DIR, "json_raw")
OUT_DIR = os.path.join(BASE_DIR, "json_processed")

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------
# 문자열 정제 함수
# ----------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    # 점선 제거
    text = re.sub(r"[·•]+", " ", text)
    text = re.sub(r"\.{2,}", " ", text)

    # 불필요한 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ----------------------------------------------------------
# 테이블 정제 함수
# ----------------------------------------------------------
def clean_table(table):
    """2차원 배열 형태 테이블에서 None → ''"""
    clean = []
    for row in table:
        clean.append([("" if c is None else str(c)) for c in row])
    return clean


# ----------------------------------------------------------
# JSON 파일 전처리
# ----------------------------------------------------------
def preprocess_json(raw_path):
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = {
        "file_name": data.get("file_name"),
        "page_count": data.get("page_count"),
        "pages": [],
    }

    for page in data.get("pages", []):
        page_num = page.get("page_number")
        text = clean_text(page.get("text", ""))

        # 테이블 정제
        tables = [clean_table(t) for t in page.get("tables", [])]

        processed["pages"].append(
            {"page_number": page_num, "clean_text": text, "clean_tables": tables}
        )

    # 저장
    out_name = os.path.splitext(os.path.basename(raw_path))[0] + "_processed.json"
    out_path = os.path.join(OUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=4)

    print(f"[완료] 전처리 JSON 저장 → {out_path}")


# ----------------------------------------------------------
# 실행부
# ----------------------------------------------------------
if __name__ == "__main__":
    json_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]

    if not json_files:
        print("[에러] json_raw 폴더에 JSON이 없습니다.")
        exit()

    for file in json_files:
        preprocess_json(os.path.join(RAW_DIR, file))
