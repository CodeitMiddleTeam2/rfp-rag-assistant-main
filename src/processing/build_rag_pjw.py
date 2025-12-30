# json으로 파일 만들기

import os
import time
import json
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
import uuid
from markdownify import markdownify as md

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# API KEY (보안상 API key 생략)
os.environ["OPENAI_API_KEY"] = "sk-..." 

# 개인에 맞게 경로 설정
BASE_DIR = r"C:\Users\USER\Desktop\AI_PROJECT\data"
CSV_PATH = os.path.join(BASE_DIR, "data_list_normalize.csv")
MD_DIR = os.path.join(BASE_DIR, "preprocessed_data") # 전처리 한 폴더로 변경
# JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "real_final_data_text.json") # 텍스트용 
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "real_final_ingest_data.json") # 임베딩용

# 임베딩 모델 설정
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 튜닝 파라미터
BATCH_SIZE = 10
CHUNK_SIZE = 1000  # 텍스트 버퍼용 제한 크기

# 날짜 변환 함수 (TIMESTAMP용)
def to_timestamp(val):
    try:
        # 데이터가 비어있거나 날짜 형식이 아닌 경우 NULL(None) 반환
        if pd.isna(val) or str(val).strip().lower() in ['nan', '', '공고서 참조', '정보없음']:
            return None
        return pd.to_datetime(val).isoformat()
    except:
        return None
    
def custom_split_logic(text, row, filename):

    chunks = []
    # 이미지 패턴 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=100)
    parts = re.split(r'(<table[\s\S]*?</table>)', text)

    # current_buffer = "" 
    current_h1 = "전체" # 대제목 (예: Ⅰ. 사업 안내)
    current_h2 = "일반" # 소제목 (예: 1. 사업 개요)

    project_name = row.get('사업명', '제목없음')


    def create_chunk_entry(content, chunk_type):
        """Supabase 스키마에 맞춘 단일 청크 생성"""

        clean_content = content.strip()

        def clean_val(key, default_val):
            val = row.get(key)
            # pd.isna(val)은 값이 NaN인지 확인하고, str(val).strip() == ""은 공백인지 확인
            if pd.isna(val) or str(val).strip().lower() == 'nan' or str(val).strip() == '':
                return default_val
            return str(val).strip()

        # 날짜 데이터
        raw_pub = row.get('공개 일자')
        raw_start = row.get('입찰 참여 시작일')
        raw_end = row.get('입찰 참여 마감일')

        raw_round = row.get('차수')
        announcement_round = int(raw_round) if pd.notna(raw_round) else 0

        # 본문과 임베딩 소스에 섹션 경로 포함
        enriched_text = f"[{current_h1} > {current_h2}]\n{clean_content}"
        embedding_input = f"사업명: {project_name} | 위치: {current_h1} > {current_h2} | 내용: {clean_content}"

        return {
            "chunk_id": str(uuid.uuid4()),
            "announcement_id": clean_val('공고 번호', '정보없음'),
            "announcement_round": announcement_round,
            "project_name": project_name,
            "project_budget": row.get('사업 금액', 0),
            "ordering_agency": clean_val('발주 기관', '미상'),

            # --- [날짜 섹션] ---
            # A. 전시용 (TEXT): "공고서 참조" 같은 글자 보존
            "published_at": clean_val('공개 일자', '정보없음'),
            "bid_start_at": clean_val('입찰 참여 시작일', '공고서 참조'),
            "bid_end_at": clean_val('입찰 참여 마감일', '공고서 참조'),
            # B. 필터링용 (TIMESTAMP): 실제 날짜 계산용
            "published_date": to_timestamp(raw_pub),
            "bid_start_date": to_timestamp(raw_start),
            "bid_end_date": to_timestamp(raw_end),

            # [본문 및 파일 정보]
            "text": enriched_text, 
            "embedding_source_text": embedding_input, # 실제 임베딩에 쓰일 텍스트
            "source_file": filename,
            "length": len(clean_content),
            "file_type": "markdown",

            # [메타데이터]
            "metadata": {
                "lang": "ko",
                "category": row.get('Category_LLM', ''),
                "depth_1": row.get('Depth_1', ''),
                "depth_2": row.get('Depth_2', ''),
                "path": [row.get('Depth_1', ''), row.get('Depth_2', '')],
                "type": chunk_type,             # header, table, text
                "parent_section": current_h1,   # 대제목
                "related_section": current_h2,  # 소제목
                "embedding_source": "project_name + sections + text"
            }
        }

    for part in parts:
        if not part.strip(): continue

        if part.strip().startswith("<table"):
            # CASE A: 표 (자르지 않고 그대로 저장)
            res = create_chunk_entry(md(part), "table")
            if res: chunks.append(res)
        else:
            # CASE B: 일반 텍스트 및 제목
            lines = part.split('\n')
            current_buffer = ""
            
            for line in lines:
                if re.match(r'^#\s', line): # 대제목
                    if current_buffer.strip():
                        for sub_txt in text_splitter.split_text(current_buffer.strip()):
                            res = create_chunk_entry(sub_txt, "text")
                            if res: chunks.append(res)
                    current_buffer = ""
                    current_h1 = line.strip().replace("#", "").strip()
                    current_h2 = "일반"
                
                elif re.match(r'^##\s', line): # 소제목
                    if current_buffer.strip():
                        for sub_txt in text_splitter.split_text(current_buffer.strip()):
                            res = create_chunk_entry(sub_txt, "text")
                            if res: chunks.append(res)
                    current_buffer = ""
                    current_h2 = line.strip().replace("##", "").strip()
                
                else:
                    current_buffer += line + "\n"
            
            if current_buffer.strip():
                for sub_txt in text_splitter.split_text(current_buffer.strip()):
                    res = create_chunk_entry(sub_txt, "text")
                    if res: chunks.append(res)
                    
    return chunks

def main():
    print(f"Ingest & Supabase JSON 생성 시작 (Batch: {BATCH_SIZE})")
    df = pd.read_csv(CSV_PATH)
    md_files_map = {f.name: f for f in list(Path(MD_DIR).rglob("*.md"))}
    final_data = []

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="전체 진행도"):
        batch_df = df.iloc[i : i + BATCH_SIZE]
        batch_chunks = []

        for _, row in batch_df.iterrows():
            filename = str(row.get('파일명'))
            if filename not in md_files_map: continue

            with open(md_files_map[filename], "r", encoding="utf-8") as f:
                batch_chunks.extend(custom_split_logic(f.read(), row, filename))

        if batch_chunks:
            # 임베딩 생성 (project_name + text 조합 사용)
            print(f"Batch {i//BATCH_SIZE + 1} 임베딩 중 ({len(batch_chunks)} 청크)")
            vectors = embeddings_model.embed_documents([c["embedding_source_text"] for c in batch_chunks])
            
            for idx, vector in enumerate(vectors):
                batch_chunks[idx]["embedding"] = vector
                # 임베딩용 임시 필드 삭제 (JSON 깔끔하게 유지)
                del batch_chunks[idx]["embedding_source_text"] 
            
            final_data.extend(batch_chunks)
        
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ [완료] - {JSON_OUTPUT_PATH}가 생성되었습니다.")

if __name__ == "__main__":
    main()

