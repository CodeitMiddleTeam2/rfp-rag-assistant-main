import os

# ===================================================================
# 프로그램 명: make_yaml_by_domain.py
# 폴더 위치: src/prompt/
# 프로그램 설명: 특정 분야별 프롬프트 초회차 생성 프로그램
# 작성이력 
#         25.12.17 한상준 최초 작성
# ===================================================================

# ==========================================
# [핵심] 스크립트가 있는 폴더 경로를 자동으로 찾습니다.
# ==========================================
# 현재 이 실행 파일의 절대 경로를 구함
current_file_path = os.path.abspath(__file__)
# 그 파일의 디렉토리(부모 폴더)만 추출 -> .../src/prompts/
target_dir = os.path.dirname(current_file_path)

print(f"📂 저장 위치가 자동으로 설정되었습니다: {target_dir}")

# ==========================================
# 1. 공통 (Common): 모든 입찰의 기본 (자격, 예산, 리스크)
# ==========================================
yaml_common = """
meta:
  domain: "Common"
  description: "모든 사업 공통 핵심 정보 (자격, 예산, 일정, 리스크)"

system_instruction_addon: |
  또한, 입찰 담당자가 '참여 여부'를 결정하는 데 치명적인(Critical) 제약 조건들을 반드시 체크해야 합니다.

user_prompt_addon: |
  ---
  [공통 추출 항목]
  RFP 내용에서 다음 정보를 포함하여 JSON을 완성해줘.
  
  1. "basic_info": {
      "budget": "사업예산 (숫자만 추출, 단위: 원)",
      "period": "사업기간 (개월 수 또는 시작~종료일)",
      "deadline": "제안서 제출 마감일시"
  }
  2. "qualifications": [
      "입찰 참가 자격 요건 리스트 (예: SI사업자 신고, 신용등급 B 이상)",
      "실적 제한 (예: 최근 3년 이내 5억 이상 실적)"
  ]
  3. "critical_risks": [
      "독소조항 또는 특이사항 (예: 지체상금률 0.25%, 저작권 발주사 귀속, 무상 유지보수 2년 등)"
  ]
"""
save_path = os.path.join(target_dir, 'extract_common.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_common)
print("✅ extract_common.yaml 생성 완료")

# ==========================================
# 2. SI (시스템 구축): 기술 스택 & 개발 환경
# ==========================================
yaml_si = """
meta:
  domain: "SI"
  description: "시스템 구축/고도화 사업"

system_prompt: |
  당신은 20년차 'SI 수석 아키텍트'입니다.
  개발팀이 이 프로젝트를 기술적으로 소화할 수 있을지 판단하기 위해, RFP에서 '기술 스택'과 '개발 환경'을 명확히 추출하세요.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 키워드에 맞는 정보를 JSON 형식으로 추출해줘. (없으면 null)

  1. "tech_stack": {
      "languages": ["Java", "Python" 등 사용 언어],
      "frameworks": ["Spring Boot", "React", "Vue" 등],
      "database": ["Oracle", "PostgreSQL" 등],
      "infrastructure": ["Linux", "Unix", "Cloud(AWS/Azure)" 등 OS/인프라]
  }
  2. "requirements": {
      "functional_count": "대략적인 기능 요구사항 개수 (또는 기능점수 FP)",
      "web_accessibility": "웹 접근성 인증 필요 여부 (boolean)",
      "security_audit": "보안 감리 수검 필요 여부 (boolean)"
  }
  3. "work_place": "근무 장소 (예: 발주사 지정 장소 상주, 원격 가능 등)"
"""
save_path = os.path.join(target_dir, 'extract_si.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_si)
print("✅ extract_si.yaml 생성 완료")

# ==========================================
# 3. SM (유지보수): 인력 & 물량
# ==========================================
yaml_sm = """
meta:
  domain: "SM"
  description: "시스템 운영/유지보수 사업"

system_prompt: |
  당신은 'IT 아웃소싱 비용 분석가'입니다.
  이 사업의 수익성(마진)을 계산하기 위해 '투입 인력'과 '관리해야 할 장비 물량'을 꼼꼼하게 파악하세요.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 정보를 JSON 형식으로 추출해줘.

  1. "manpower_structure": {
      "total_count": "필수 상주 인원 수 (명)",
      "roles": ["PM(고급)", "PL(중급)", "운영(초급)" 등 요구 등급 분포],
      "dispatch_type": "상주 / 비상주 / 혼합"
  }
  2. "target_assets": {
      "hardware_count": "관리 대상 서버/네트워크 장비 수량 요약",
      "software_count": "관리 대상 상용 SW/DBMS 수량 요약",
      "users": "시스템 사용자 수 (규모 파악용)"
  }
  3. "service_level": {
      "sla_applied": "SLA(서비스수준협약) 적용 여부 (boolean)",
      "night_shift": "야간/휴일 장애 대응 필요 여부 (boolean)"
  }
"""
save_path = os.path.join(target_dir, 'extract_sm.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_sm)
print("✅ extract_sm.yaml 생성 완료")


# ==========================================
# 4. ISP (컨설팅): 현황 & 방향성
# ==========================================
yaml_isp = """
meta:
  domain: "ISP"
  description: "정보화전략계획/컨설팅 사업"

system_prompt: |
  당신은 'DX(디지털전환) 전문 컨설턴트'입니다.
  이 컨설팅 용역의 난이도를 파악하기 위해, 분석해야 할 '현행 범위'와 고객이 원하는 '미래 모습'을 추출하세요.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 정보를 JSON 형식으로 추출해줘.

  1. "scope": {
      "as_is_analysis": "현황 분석 대상 업무/시스템 범위 요약",
      "to_be_design": "목표 모델 설계 범위 (예: 차세대 시스템 구축, 클라우드 전환 로드맵 등)"
  }
  2. "constraints": {
      "consultants": "투입 컨설턴트 자격 요건 (예: 정보시스템감리사, 기술사 등)",
      "methodology": "특정 방법론(Methodology) 적용 요구 여부"
  }
  3. "deliverables": ["정보화전략계획서", "BPR 산출물", "RFP 초안 작성" 등 주요 산출물 목록]
"""
save_path = os.path.join(target_dir, 'extract_isp.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_isp)
print("✅ extract_isp.yaml 생성 완료")


# ==========================================
# 5. Infra (인프라/보안): 장비 스펙
# ==========================================
yaml_infra = """
meta:
  domain: "Infra"
  description: "서버/네트워크/보안장비 도입 사업"

system_prompt: |
  당신은 '하드웨어/인프라 엔지니어'입니다.
  납품해야 할 장비의 구체적인 '스펙(Spec)'과 '수량'을 파악하는 것이 가장 중요합니다.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 정보를 JSON 형식으로 추출해줘.

  1. "equipment_list": [
      {"item": "품목명(예: x86서버)", "spec": "주요 스펙(CPU, RAM 등)", "count": "수량"}
  ]
  2. "installation": {
      "location": "납품 및 설치 장소",
      "migration": "기존 데이터 이관(Migration) 필요 여부 (boolean)"
  }
  3. "warranty": "무상 하자보수 기간 (예: 1년, 3년)"
"""
save_path = os.path.join(target_dir, 'extract_infra.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_infra)
print("✅ extract_infra.yaml 생성 완료")


# ==========================================
# 6. Data (데이터/AI): 데이터 양 & 품질
# ==========================================
yaml_data = """
meta:
  domain: "Data"
  description: "DB구축/AI학습용데이터/라벨링 사업"

system_prompt: |
  당신은 'AI 데이터 엔지니어'입니다.
  데이터 구축 사업의 견적은 '데이터의 양(Volume)'과 '난이도(가공 수준)'에서 나옵니다. 이를 중점적으로 추출하세요.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 정보를 JSON 형식으로 추출해줘.

  1. "data_scope": {
      "source_type": "원천 데이터 유형 (텍스트, 이미지, 음성, 영상 등)",
      "target_volume": "구축 목표 수량 (예: 10만 건, 5,000 시간)",
      "process_level": "가공 수준 (단순 정제, 라벨링, 메타데이터 추출 등)"
  }
  2. "quality_goal": "목표 품질 지표 (예: 정확도 99% 이상, 검수율 100%)"
  3. "tool": "저작도구(Tool) 개발 필요 여부 (boolean)"
"""
save_path = os.path.join(target_dir, 'extract_data.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_data)
print("✅ extract_data.yaml 생성 완료")


# ==========================================
# 7. Web (홈페이지): 디자인 & 콘텐츠
# ==========================================
yaml_web = """
meta:
  domain: "Web"
  description: "홈페이지/포털 구축 및 디자인 사업"

system_prompt: |
  당신은 '웹 에이전시 PM'입니다.
  디자인 시안 개수와 콘텐츠 이관 물량, 웹 표준 준수 여부가 견적의 핵심입니다.

user_prompt_template: |
  다음 RFP 내용을 분석하여 아래 정보를 JSON 형식으로 추출해줘.

  1. "design_scope": {
      "device": "지원 기기 (PC, Mobile, Tablet, 반응형 등)",
      "template_count": "요구되는 메인/서브 디자인 시안 개수"
  }
  2. "content_migration": "기존 게시물/콘텐츠 이관 물량 (건수)"
  3. "compliance": ["웹 접근성 인증", "웹 표준 준수", "개인정보보호 심사" 등 준수 사항]
"""
save_path = os.path.join(target_dir, 'extract_web.yaml')
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(yaml_web)
print("✅ extract_web.yaml 생성 완료")

print("✅ [성공] 6대 도메인 + 공통 프롬프트(YAML) 파일 생성을 완료했습니다.")
print(f"\n🎉 모든 파일이 '{target_dir}' 폴더 안에 안전하게 생성되었습니다!")