import os
import pandas as pd
import olefile
import zlib
import struct
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader

# ===================================================================
# 프로그램 명: classify_by_content.py
# 위치: src/prompt/classify_by_content.py
# 프로그램 설명: 원본 pdf/hwp 데이터의 본문을 보고 도메인 카테고리 분류, 이후 csv파일로 결과물 저장
# 작성이력 
#         25.12.17 한상준 최초 작성
# ===================================================================

# --- 설정 ---
load_dotenv()
current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent
DATA_DIR = PROJECT_ROOT / "data/rfp_data"
OUTPUT_FILE = "rfp_classification_precise.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------------------------------------------
# [핵심] HWP 본문 강제 추출 함수 (Deep Extraction)
# ----------------------------------------------------------------
def extract_hwp_text_deep(file_path):
    try:
        if not olefile.isOleFile(file_path):
            return "HWP 포맷 아님 (HWPX일 가능성 있음)"

        f = olefile.OleFileIO(file_path)
        dirs = f.listdir()
        
        # 1. BodyText 섹션 찾기 (본문 내용이 담긴 곳)
        # 보통 BodyText/Section0, Section1... 형태로 존재함
        body_sections = [d for d in dirs if d[0] == "BodyText"]
        
        extracted_text = ""
        
        for section in body_sections:
            stream = f.openstream(section)
            data = stream.read()
            
            # 2. Zlib 압축 해제 (HWP 본문은 압축되어 있음)
            try:
                # -15: raw stream (header 없이 압축된 데이터 처리)
                decompressed = zlib.decompress(data, -15) 
            except:
                try:
                    decompressed = zlib.decompress(data)
                except:
                    continue # 압축 해제 실패 시 다음 섹션으로
            
            # 3. 텍스트 변환 (UTF-16LE)
            # HWP 텍스트는 유니코드(UTF-16 Little Endian)로 저장됨
            section_text = decompressed.decode('utf-16le', errors='ignore')
            
            # 4. 제어 문자 및 불필요한 태그 제거 (간단한 정제)
            # HWP 특수문자나 표 제어문자 등이 섞여있으므로, 일반 텍스트만 필터링
            clean_text = "".join([c for c in section_text if c.isprintable() or c in ['\n', ' ', '\t']])
            extracted_text += clean_text + "\n"
            
            # 앞부분 4000자만 모으면 충분함 (메타데이터 분석용)
            if len(extracted_text) > 4000:
                break
                
        f.close()
        
        if not extracted_text.strip():
            return "본문 추출 실패 (암호화 또는 빈 문서)"
            
        return extracted_text[:4000]

    except Exception as e:
        return f"Error: {str(e)}"

# --- PDF 텍스트 추출 ---
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        # 넉넉하게 7페이지까지 읽기
        for page in reader.pages[:min(7, len(reader.pages))]:
            text += page.extract_text() or ""
        return text[:4000]
    except Exception as e:
        return ""

# --- LLM 분류기 (동일) ---
def classify_file_content(filename, content):
    if len(content) < 50:
        return "판독불가"

    prompt = f"""
    당신은 프로젝트 제안요청서(RFP) 분류 전문가입니다.
    제공된 [파일명]과 [문서 내용]을 분석하여, 이 사업의 성격에 가장 부합하는 카테고리를 하나만 선택하세요.

    [분류 카테고리]
    1. IT_정보화: 소프트웨어 개발, 시스템 구축, 통신망, 전산 장비(서버/PC) 도입 등
    2. 공사_시설: 건축, 토목, 인테리어, 전기/소방 공사, 시설물 설치 등
    3. 물품_구매: 가구, 차량, 의약품, 일반 비품, 기자재 단순 구매 (IT 장비 제외)
    4. 용역_일반: 학술 연구, 행사 대행, 청소/경비, 홍보물 제작, 번역, 단순 인력 파견 등

    [파일명]: {filename}
    [내용]: {content}

    [지침]
    1. 내용이 복합적일 경우, 예산 비중이 더 크거나 주된 과업이라고 판단되는 쪽으로 분류하세요.
    2. 출력은 오직 위 4개 중 해당하는 '카테고리명' 하나만 반환하세요. (예: 공사_시설)
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --- 메인 실행 ---
if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.hwp', '.pdf'))]
    results = []
    
    print(f"🕵️ 총 {len(files)}개 파일의 심층 분석(Deep Analysis)을 시작합니다...")

    for i, fname in enumerate(files):
        file_path = DATA_DIR / fname
        content = ""
        
        # 1. 텍스트 추출
        if fname.lower().endswith('.hwp'):
            content = extract_hwp_text_deep(file_path)
        elif fname.lower().endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
        
        # 2. 내용 확인 및 분류
        if len(content) < 20 or "Error" in content or "실패" in content:
            print(f"[{i+1}/{len(files)}] ⚠️ {fname}: {content[:30]}...")
            category = "판독불가"
        else:
            # LLM 분류 실행
            category = classify_file_content(fname, content)
            print(f"[{i+1}/{len(files)}] ✅ {fname} -> {category}")

        results.append({
            "FileName": fname, 
            "Category": category, 
            "ExtractedSnippet": content[:100].replace('\n', ' ')
        })

    # 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 분석 완료! '{OUTPUT_FILE}' 파일을 확인하세요.")
    print("📊 결과 요약:")
    print(df['Category'].value_counts())