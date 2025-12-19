import pandas as pd

# 1. 데이터 로드 (가장 깨끗한 원본 데이터인 data_list.csv 사용)
# 파일 경로가 다르다면 이 부분을 수정해주세요.
df = pd.read_csv('./data/data_list.csv')

# ===================================================================
# 프로그램 명: classify_metadata.py
# 폴더 위치: src/prompt/
# 프로그램 설명: 메타데이터 csv 파일을 읽고 키워드 기반으로 데이터 카테고리를 분류
# 작성이력 
#         25.12.17 한상준 최초 작성
# ===================================================================

# ==========================================
# [작성 의도]
# 사용자가 지정한 4가지 명확한 카테고리(IT, 공사, 물품, 용역)에 맞춰
# 키워드 기반의 조건문(Rule-based)을 사용하여 데이터를 분류합니다.
# 상위 조건(IT)부터 순차적으로 검사하여 중복을 방지합니다.
# ==========================================

def classify_user_defined(row):
    # 제목과 본문을 합쳐서 검사 (정확도 향상)
    # 제목의 띄어쓰기를 없애서 매칭 확률을 높입니다 (예: '유지 보수' -> '유지보수')
    title = str(row['사업명']).replace(" ", "") 
    body = str(row['텍스트'])
    
    # 1순위: IT_정보화
    # (소프트웨어, 시스템, 통신망, 서버, AI 등)
    keywords_it = ['정보화', '시스템', '소프트웨어', 'SW', '플랫폼', '홈페이지', '웹', '앱', 
                   '서버', '네트워크', '통신망', '전산', 'ERP', 'LMS', '클라우드', '빅데이터', 
                   '유지보수', '데이터구축', '기능개선', '고도화', '솔루션']
    
    if any(k in title for k in keywords_it): 
        return 'IT_정보화'

    # 2순위: 공사_시설
    # (건축, 토목, 전기, 인테리어, 시설물 등)
    keywords_construct = ['공사', '건설', '건축', '토목', '인테리어', '리모델링', '전기', 
                          '소방', '설비', '시공', '증축', '개보수', '철거', '배관']
    
    if any(k in title for k in keywords_construct):
        return '공사_시설'

    # 3순위: 물품_구매
    # (IT 장비를 제외한 가구, 차량, 의약품 등)
    keywords_goods = ['구매', '구입', '납품', '차량', '가구', '의약품', '기자재', '비품', '장비', '도서']
    
    if any(k in title for k in keywords_goods):
        return '물품_구매'

    # 4순위: 용역_일반
    # (학술, 행사, 청소, 홍보, 번역 등)
    keywords_service = ['용역', '연구', '조사', '학술', '행사', '대행', '청소', '경비', 
                        '홍보', '디자인', '콘텐츠', '번역', '파견', '위탁', '컨설팅', '양성', '교육']
    
    if any(k in title for k in keywords_service):
        return '용역_일반'

    # 5순위: 기타
    # 위 어디에도 속하지 않는 경우
    return '기타'

# 2. 분류 적용
print("🚀 분류를 시작합니다...")
df['Category'] = df.apply(classify_user_defined, axis=1)

# 3. 결과 확인 (통계)
print("\n📊 [분류 결과 통계]")
print(df['Category'].value_counts())

# 4. 파일 저장
save_name = 'metadata_added_category.csv'
df.to_csv(save_name, index=False, encoding='utf-8-sig')
print(f"\n💾 '{save_name}' 파일로 저장 완료!")