import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# ===================================================================
# 프로그램 명: classify_data_by_domain
# 폴더 위치: src/prompt/classify_data_by_domain.py
# 프로그램 설명: 원본 pdf/hwp 데이터의 제목을 보고 도메인 카테고리 분류, 이후 root 디렉토리에 csv파일로 결과물 저장
# 작성이력 
#         25.12.17 한상준 최초 작성
# ===================================================================


# ----------------------------------------------------------------
# 1. 설정
# ----------------------------------------------------------------
current_dir = Path(__file__).resolve().parent # root/src/prompt
project_root = current_dir.parent.parent # root

DATA_DIR = project_root / "data/rfp_data"  # RFP 파일들이 들어있는 폴더 경로
OUTPUT_FILE = "rfp_classification_result.csv"

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("API Key가 없습니다. .env 파일을 확인해주세요!")
else:
    print(f"API Key가 로드되었습니다. (시작: {API_KEY[:5]}...)")

client = OpenAI(api_key=API_KEY)

# ----------------------------------------------------------------
# 2. 파일 목록 읽어오기
# ----------------------------------------------------------------
def get_file_list(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.hwp', '.pdf', '.hwpx'))]
    print(f"📂 총 {len(files)}개의 파일을 발견했습니다.")
    return files

# ----------------------------------------------------------------
# 3. LLM을 이용한 분류 로직
# ----------------------------------------------------------------
def classify_files_by_name(file_names):
    # 한 번에 너무 많은 파일을 보내면 토큰 제한에 걸릴 수 있으므로 50개씩 나눠서 보냄
    batch_size = 50
    results = []

    prompt_template = """
    아래는 공공/기업 프로젝트의 제안요청서(RFP) 파일명 목록입니다. 
    각 파일명을 보고 가장 적절한 '사업 유형(Category)'을 다음 중에서 하나만 선택하여 분류하세요.

    [분류 카테고리]
    1. SI_구축 (시스템 개발, 고도화, 차세대 등)
    2. SM_운영 (유지보수, 운영지원, 위탁운영 등)
    3. H/W_구매 (서버, 스토리지, PC 도입 등)
    4. S/W_구매 (라이선스 구입, 패키지 도입 등)
    5. 컨설팅_ISP (정보화전략계획, BPR, 감리 등)
    6. 기타 (분류가 모호한 경우)

    [출력 형식]
    파일명1: 카테고리명
    파일명2: 카테고리명
    ...

    [대상 파일 목록]
    {files}
    """

    for i in range(0, len(file_names), batch_size):
        batch = file_names[i:i+batch_size]
        file_list_str = "\n".join(batch)
        
        print(f"🤖 {i+1}~{min(i+batch_size, len(file_names))}번째 파일 분류 중...")
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "당신은 IT 프로젝트 분류 전문가입니다."},
                {"role": "user", "content": prompt_template.format(files=file_list_str)}
            ]
        )
        
        # 응답 파싱 (간단한 문자열 처리)
        lines = response.choices[0].message.content.strip().split('\n')
        for line in lines:
            if ":" in line:
                fname, category = line.split(":", 1)
                results.append({"FileName": fname.strip(), "Category": category.strip()})

    return results

# ----------------------------------------------------------------
# 4. 실행 및 저장
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 파일명 로드
    files = get_file_list(DATA_DIR)
    
    if files:
        # 분류 실행
        classified_data = classify_files_by_name(files)
        
        # 결과 저장
        df = pd.DataFrame(classified_data)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') # 한글 깨짐 방지
        
        print(f"\n✅ 분류 완료! '{OUTPUT_FILE}' 파일에 저장되었습니다.")
        print("\n📊 카테고리별 분포:")
        print(df['Category'].value_counts())
    else:
        print("❌ 처리할 파일이 없습니다.")