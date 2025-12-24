from llama_cpp import Llama

#==============================================
# 프로그램명: load_local_model.py
# 폴더위치: src/generation/load_local_model.py
# 프로그램 설명: 학습된 로컬 모델을 불러오는 모듈 함수
# 작성이력: 25.12.23 한상준 최초 작성
#===============================================

# 모델 로드

def load_model():
    return Llama(
        model_path="../unsloth.Q4_K_M.gguf",  # 방금 만든 모델 파일 경로
        n_gpu_layers=-1,      # L4 GPU를 100% 활용 (모든 레이어 GPU 로드)
        n_ctx=4096,           # 컨텍스트 길이 (RFP 문서가 길 수 있으니 넉넉하게)
        verbose=True          # 터미널에 로그 출력
    )