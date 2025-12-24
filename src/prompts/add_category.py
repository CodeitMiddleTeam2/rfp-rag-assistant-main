import pandas as pd
import os

#==============================================
# 프로그램명: add_category.py
# 폴더위치: src/prompts/add_category.py
# 프로그램 설명: 원본 메타데이터 csv 파일에 카테고리 분류를 추가하는 프로그램
# 작성이력: 25.12.17 한상준 최초 작성
#===============================================

# [설계 의도]
# 데이터 팀의 결과물이 오기 전, UI 테스트를 위해 
# 대분류(Depth_1)는 기존 LLM 결과를 활용하고, 
# 중분류(Depth_2)는 사업명의 핵심 키워드를 기반으로 임시 생성합니다.

def generate_mock_hierarchy():
    file_path = 'final_classification_llm.csv'
    if not os.path.exists(file_path):
        print(f"❌ {file_path} 파일이 없습니다.")
        return

    df = pd.read_csv(file_path)

    # 1. Depth_1 (대분류): LLM이 분류한 Category 활용
    df['Depth_1'] = df['Category_LLM']

    # 2. Depth_2 (중분류) 생성 로직
    def classify_depth2(row):
        title = str(row['사업명']).replace(" ", "")
        d1 = row['Depth_1']
        
        # IT_정보화 세부 분류
        if d1 == 'IT_정보화':
            if any(k in title for k in ['유지관리', '유지보수', '운영', '위탁']): return '운영 및 유지관리'
            if any(k in title for k in ['ISP', 'BPR', '전략수립', '마스터플랜']): return '전략 컨설팅'
            if any(k in title for k in ['DB', '데이터', '빅데이터', 'AI', '인공지능']): return '데이터 및 AI'
            if any(k in title for k in ['인프라', '서버', '네트워크', '장비']): return '인프라 도입'
            return '시스템 구축 및 고도화' # 기본값
        
        # 비 IT 도메인 분류
        elif d1 == '용역_일반':
            if any(k in title for k in ['연구', '조사', '분석']): return '연구 및 조사'
            return '일반 행정 용역'
        
        elif d1 == '공사_시설':
            return '시설 시공'
        
        elif d1 == '물품_구매':
            return '기자재 구입'
            
        return '기타 상세'

    df['Depth_2'] = df.apply(classify_depth2, axis=1)

    # 3. 결과 저장
    save_path = 'final_classification_hierarchy.csv'
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 계층 데이터 생성 완료: {save_path}")
    
    # 분포 확인
    print("\n📊 [Depth_2 분포 현황]")
    print(df['Depth_2'].value_counts())

if __name__ == "__main__":
    generate_mock_hierarchy()