import pandas as pd
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 데이터 로드
df = pd.read_csv('./metadata_added_category.csv')

# ==========================================
# [설계 의도]
# 키워드 매칭의 한계(중의적 단어, 문맥 무시)를 극복하기 위해
# LLM(GPT)에게 '분류 전문가' 페르소나를 부여하여 의미 기반으로 판단하게 합니다.
# ==========================================

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("API Key가 없습니다. .env 파일을 확인해주세요!")
else:
    print(f"API Key가 로드되었습니다. (시작: {API_KEY[:5]}...)")

client = OpenAI(api_key=API_KEY)

def classify_by_llm(row):
    title = row['사업명']
    summary = str(row['텍스트'])[:500]  # 비용 절감을 위해 앞 500자만
    
    prompt = f"""
    당신은 B2G 공공입찰 사업 분류 전문가입니다.
    아래 사업의 제목과 내용을 읽고, 가장 적합한 카테고리를 하나만 선택하여 반환하세요.
    
    [카테고리 목록]
    1. IT_정보화 (시스템 구축, SW개발, DB, 서버/네트워크, 유지보수 등)
    2. 공사_시설 (건축, 토목, 전기, 통신공사, 인테리어, 설비 등)
    3. 물품_구매 (단순 물품/기자재 구입, 상용SW/장비 납품 등)
    4. 용역_일반 (단순 인력 파견, 행사, 학술연구, 청소, 번역 등)
    5. 기타
    
    [판단 기준]
    - '시스템', '구축' 등의 단어가 있어도 실제 내용이 '전기 공사'라면 '공사_시설'로 분류하세요.
    - '구매'라고 되어 있어도 '서버/SW' 도입이 핵심이면 'IT_정보화'로 분류하세요.
    
    [입력 데이터]
    사업명: {title}
    내용요약: {summary}
    
    [출력 형식]
    카테고리명만 딱 출력하세요. (예: IT_정보화)
    """
    
    try:
        # 실제 API 호출
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "정확한 카테고리 분류기입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# 2. 적용 (실제 실행 시 주석 해제)
print("🧠 LLM이 의미 기반으로 정밀 분류를 시작합니다...")
df['Category_LLM'] = df.apply(classify_by_llm, axis=1)

# 3. 결과 비교 (기존 키워드 방식 vs LLM 방식)
print(df[['사업명', 'Category', 'Category_LLM']].head(10))

# 4. 저장
df.to_csv('final_classification_llm.csv', index=False, encoding='utf-8-sig')