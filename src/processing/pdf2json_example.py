import os
import json
import pdfplumber

BASE_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data"
PDF_DIR = os.path.join(BASE_DIR, "output", "pdf")
JSON_DIR = os.path.join(BASE_DIR, "output", "json")
os.makedirs(JSON_DIR, exist_ok=True)


def extract_pdf_to_json(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    output_name = os.path.splitext(pdf_name)[0] + ".json_raw"
    output_path = os.path.join(JSON_DIR, output_name)

    result = {"file_name": pdf_name, "page_count": 0, "pages": []}

    with pdfplumber.open(pdf_path) as pdf:
        result["page_count"] = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            # 테이블 추출
            tables = []
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                tables.append(table)

            result["pages"].append(
                {
                    "page_number": page_num,
                    "text": page_text,
                    "tables": tables,
                }
            )

    # JSON 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"[완료] JSON 생성 → {output_path}")


# -------------------------------------------------------
# PDF 폴더 내 모든 파일 처리 + 진행률 표시
# -------------------------------------------------------
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    total = len(pdf_files)
    if total == 0:
        print("[오류] PDF 폴더에 PDF 파일이 없습니다.")
        exit()

    print(f"총 {total}개 PDF 파일 처리 시작...\n")

    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"[진행] {idx}/{total} → 처리중: {pdf_file}")

        pdf_path = os.path.join(PDF_DIR, pdf_file)
        extract_pdf_to_json(pdf_path)

    print(f"\n🎉 모든 PDF({total}개) 처리 완료!")
