import os
import re
import torch
import gc
import streamlit as st
from openai import OpenAI
from langsmith import traceable

#==============================================
# 프로그램명: model_manager.py
# 폴더위치: src/generation/model_manager.py
# 프로그램 설명: 웹 데모에서 모델을 선택하게끔(로컬 or API) 만들어주는 매니저 클래스
# 작성이력: 25.12.23 한상준 최초 작성
# 25.12.29 정규표현식 전처리 추가
# 25.12.29 LangSmith 추적 추가
#===============================================

# 캐싱할 함수는 클래스 밖(또는 staticmethod)에 정의합니다.
@st.cache_resource
def _load_llama_cpp_model(model_path: str, n_ctx: int = 24576):
    """
    실제 무거운 모델 로딩을 수행하는 함수입니다.
    이 함수의 반환값(Llama 객체)이 캐싱됩니다.
    """
    try:
        from llama_cpp import Llama
        # print(f"📂 로컬 모델 로딩 시작: {model_path}") # 디버깅용
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # L4 GPU 활용
            n_ctx=n_ctx,
            verbose=True,
        )
        return llm
    except Exception as e:
        st.error(f"❌ Llama 모델 초기화 실패: {e}")
        return None

class ModelManager:
    def __init__(self, local_model_path: str = "../unsloth.Q4_K_M.gguf"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.local_model_path = local_model_path

    def get_openai_client(self):
        """OpenAI 클라이언트 반환 (가벼운 객체라 캐싱 불필요)"""
        if not self.api_key:
            st.error("🚨 .env 파일에 OPENAI_API_KEY가 없습니다.")
            st.stop()
        return OpenAI(api_key=self.api_key)

    def load_local_model(self):
        """
        클래스 메서드는 단순히 캐싱된 전역 함수를 호출하는 역할만 합니다.
        """
        if not os.path.exists(self.local_model_path):
            st.error(f"🚨 모델 파일이 없습니다: {self.local_model_path}")
            return None
            
        # 여기서 캐싱된 함수 호출 (_load_llama_cpp_model)
        return _load_llama_cpp_model(self.local_model_path)

    @traceable(run_type="llm", name="LLM_Generation")
    def generate_response(self, messages, source="openai", local_llm=None, openai_client=None):
        """답변 생성 로직 통합"""
        try:
            if source == "openai":
                if not openai_client:
                    return "🚨 OpenAI Client가 연결되지 않았습니다."
                
                response = openai_client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=messages
                )
                return response.choices[0].message.content
            
            elif source == "local":
                if not local_llm:
                    return "🚨 로컬 모델이 로드되지 않았습니다."
                
                # 로컬 모델 추론
                response = local_llm.create_chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    stop=["<|im_end|>", "<|endoftext|>", "User:"],
                    temperature=0.1
                )
                raw_content = response['choices'][0]['message']['content']

                # ✅ [핵심 수정] <think> ... </think> 태그 제거 로직
                # re.DOTALL: 줄바꿈이 포함된 내용도 모두 찾음
                clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                
                return clean_content
                
        except Exception as e:
            return f"❌ 답변 생성 중 에러 발생: {str(e)}"

    def clear_gpu_memory(self):
        """GPU 메모리 캐시를 강제로 비웁니다."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # print("🧹 GPU 메모리 캐시 초기화 완료")