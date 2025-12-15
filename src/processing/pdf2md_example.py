import os
import pdfplumber

BASE_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data"
PDF_DIR = os.path.join(BASE_DIR, "output", "pdf")
MD_DIR = os.path.join(BASE_DIR, "output", "md")
os.makedirs(MD_DIR, exist_ok=True)


def table_to_markdown(table):
    """
    2차원 리스트 형태의 table 데이터를 Markdown 표로 변환
    (None → "" 처리하여 오류 방지)
    """
    if not table or len(table) == 0:
        return ""

    md = []

    # 첫 행을 header로 변환
    header = [str(cell) if cell is not None else "" for cell in table[0]]

    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")

    # 나머지 행 처리
    for row in table[1:]:
        safe_row = [str(cell) if cell is not None else "" for cell in row]
        md.append("| " + " | ".join(safe_row) + " |")

    return "\n".join(md)


def extract_pdf_to_md(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    output_name = os.path.splitext(pdf_name)[0] + ".md"
    output_path = os.path.join(MD_DIR, output_name)

    md_lines = []
    md_lines.append(f"# 📄 {pdf_name}\n")

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        md_lines.append(f"총 **{page_count} 페이지**\n\n")

        for page_num, page in enumerate(pdf.pages, start=1):
            md_lines.append(f"---\n\n## 📘 페이지 {page_num}\n")

            # 텍스트 추출
            text = page.extract_text() or ""
            md_lines.append("### 📌 텍스트\n")
            md_lines.append("```\n" + text + "\n```\n")

            # 테이블 추출
            tables = page.extract_tables()
            for idx, table in enumerate(tables):
                md_lines.append(f"### 📊 표 {idx+1}\n")
                table_md = table_to_markdown(table)
                md_lines.append(table_md + "\n")

    # Markdown 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[완료] Markdown 저장됨 → {output_path}")


# -------------------------------------------------------
# PDF 한 개만 처리
# -------------------------------------------------------
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("[오류] PDF 폴더에 PDF 파일이 없습니다.")
        exit()

    # 첫 번째 PDF만 선택
    first_pdf = pdf_files[0]
    pdf_path = os.path.join(PDF_DIR, first_pdf)

    print(f"[처리중] {pdf_path}")
    extract_pdf_to_md(pdf_path)
