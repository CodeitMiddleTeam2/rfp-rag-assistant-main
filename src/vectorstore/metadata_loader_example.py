import pandas as pd
import os

EXCEL_PATH = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data\data_list.xlsx"

COLUMN_MAPPING = {
    "공고 번호": "notice_id",
    "공고 차수": "notice_round",
    "사업명": "project_name",
    "사업 금액": "project_budget",
    "발주 기관": "ordering_agency",
    "공개 일자": "publish_date",
    "입찰 참여 시작일": "bid_start_date",
    "입찰 참여 마감일": "bid_end_date",
    "사업 요약": "project_summary",
    "파일형식": "file_type",
    "파일명": "file_name",
}


def load_metadata_map():
    """
    Excel(data_list.xlsx)을 읽어서
    {파일명(확장자 제거): 메타 dict} 형태로 반환
    """
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel 파일이 없습니다: {EXCEL_PATH}")

    df = pd.read_excel(EXCEL_PATH)
    meta_map = {}

    for _, row in df.iterrows():
        raw_file_name = str(row.get("파일명", "")).strip()
        if not raw_file_name:
            continue

        file_key = os.path.splitext(raw_file_name)[0]

        meta = {}
        for kor, eng in COLUMN_MAPPING.items():
            v = row.get(kor)
            meta[eng] = "" if pd.isna(v) else str(v)

        # RAG에서 유용한 고정 필드
        meta["document_id"] = meta.get("notice_id")

        meta_map[file_key] = meta

    return meta_map
