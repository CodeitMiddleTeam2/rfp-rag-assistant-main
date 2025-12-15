import os
import pdfplumber

BASE_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data"
PDF_DIR = os.path.join(BASE_DIR, "output", "pdf")
HTML_DIR = os.path.join(BASE_DIR, "output", "html")
os.makedirs(HTML_DIR, exist_ok=True)


def extract_pdf_to_html(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    output_name = os.path.splitext(pdf_name)[0] + ".html"
    output_path = os.path.join(HTML_DIR, output_name)

    html_parts = []

    # HTML 기본 스타일
    html_parts.append(
        """
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .page { margin-bottom: 40px; padding: 20px; border: 1px solid #ccc; }
            .page h2 { border-bottom: 1px solid #aaa; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            table, th, td { border: 1px solid #444; }
            th, td { padding: 8px; }
        </style>
    </head>
    <body>
    """
    )

    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

        html_parts.append(f"<h1>📄 {pdf_name} (총 {page_count} 페이지)</h1>")

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()

            html_parts.append(f"<div class='page'>")
            html_parts.append(f"<h2>페이지 {page_num}</h2>")

            # 텍스트 영역
            html_parts.append("<pre>")
            html_parts.append(text)
            html_parts.append("</pre>")

            # 테이블 영역
            for idx, table in enumerate(tables):
                html_parts.append(f"<h3>표 {idx+1}</h3>")
                html_parts.append("<table>")

                for row in table:
                    html_parts.append("<tr>")
                    for cell in row:
                        html_parts.append(f"<td>{cell}</td>")
                    html_parts.append("</tr>")

                html_parts.append("</table>")

            html_parts.append("</div>")  # page end

    # HTML 닫기
    html_parts.append("</body></html>")

    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"[완료] HTML 저장됨 → {output_path}")


# -------------------------------------------------------
# PDF 1개만 처리
# -------------------------------------------------------
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("[오류] PDF 폴더에 PDF 파일이 없습니다.")
        exit()

    # 첫 번째 PDF 파일만 처리
    first_pdf = pdf_files[0]
    pdf_path = os.path.join(PDF_DIR, first_pdf)

    print(f"[처리중] {pdf_path}")
    extract_pdf_to_html(pdf_path)
