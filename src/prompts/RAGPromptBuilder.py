import os
import yaml

class RAGPromptBuilder:
    def __init__(self, prompt_dir):
        """
        [작성 의도]
        객체 생성 시 프롬프트 파일이 저장된 경로를 고정하고, 
        대화의 기본 골격인 rag_chat_core.yaml을 미리 로드하여 성능을 최적화합니다.
        """
        self.prompt_dir = os.path.dirname(os.path.abspath(__file__))
        self.core_path = os.path.join(prompt_dir, 'rag_chat_core.yaml')
        
        with open(self.core_path, 'r', encoding='utf-8') as f:
            self.core = yaml.safe_load(f)

    def _determine_yaml(self, llm_category, title):
        """
        [작성 의도: 2단계 라우팅]
        - 1단계: LLM이 판단한 대분류(Category_LLM)를 기준으로 비IT 사업을 먼저 걸러냅니다.
        - 2단계: IT 사업인 경우에만 제목 키워드를 분석하여 SI/SM/ISP 등의 세부 프롬프트를 선택합니다.
        
        [작용]
        불필요한 기술 질문이 비IT 사업(예: 청소, 가구 구매)에 던져지는 것을 방지하여 환각을 억제합니다.
        """
        cat = str(llm_category).strip()
        title_clean = str(title).replace(" ", "")

        # 1. 비IT 도메인인 경우 -> 전용 프롬프트가 없으면 공용(Common) 사용
        # 현재 MVP 전략에 따라 비IT는 extract_common.yaml로 일원화 대응 가능
        if cat != 'IT_정보화':
            return 'extract_common.yaml'

        # 2. IT_정보화인 경우 세부 라우팅
        if any(k in title_clean for k in ['ISP', 'ISMP', '전략', '컨설팅']):
            return 'extract_isp.yaml'
        elif any(k in title_clean for k in ['유지보수', '운영', '위탁', '관리']):
            return 'extract_sm.yaml'
        elif any(k in title_clean for k in ['DB', '데이터', 'AI', '인공지능', '빅데이터']):
            return 'extract_data.yaml'
        elif any(k in title_clean for k in ['서버', '네트워크', '보안', '인프라', '장비']):
            return 'extract_infra.yaml'
        elif any(k in title_clean for k in ['홈페이지', '웹', '디자인', 'UI', 'UX']):
            return 'extract_web.yaml'
        
        # 위 조건에 해당하지 않는 일반적인 IT 사업은 SI로 간주
        return 'extract_si.yaml'

    def _get_persona(self, yaml_filename):
        """
        [작용] 
        선택된 YAML 파일에서 'system_prompt'의 첫 줄(역할 정의)만 추출하여 
        대화형 시스템 프롬프트에 주입합니다.
        """
        path = os.path.join(self.prompt_dir, yaml_filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # "당신은 ~입니다." 부분만 가져오기
                return config.get('system_prompt', "B2G 입찰 분석 전문가").split('\n')[0]
        except:
            return "B2G 공공입찰 컨설팅 AI"

    def build_messages(self, category, title, context, history, query):
        """
        [작성 의도] app.py로부터 데이터를 받아 LLM에 전송할 최종 메시지 리스트를 조립합니다.
        [인자] category, title, context, history, query (app.py의 키워드와 1:1 매칭)
        """
        # 1. 적절한 도메인 YAML 파일 선택
        target_file = self._determine_yaml(category, title)
        
        # 2. 페르소나(역할) 획득
        role = self._get_persona(target_file)
        
        # 3. 시스템 프롬프트 조립
        system_msg = self.core['system_prompt_template'].format(
            domain=target_file.replace('extract_', '').replace('.yaml', '').upper(),
            role=role
        )
        
        # 4. 대화 내역 포맷팅 (최근 5턴 유지)
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]]) if history else "없음"
        
        # 5. 사용자 프롬프트 조립 (원본 텍스트가 길 경우를 대비해 4000자로 슬라이싱)
        user_msg = self.core['user_prompt_template'].format(
            context=str(context)[:4000], 
            history=history_str,
            query=query
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]