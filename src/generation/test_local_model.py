import os
from llama_cpp import Llama
import time

# 1. 모델 경로 설정 (정확한지 확인!)
# 같은 폴더에 모델이 있다고 가정합니다. 다르다면 절대 경로를 입력하세요.
MODEL_PATH = r"/home/spai0525/rfp-rag-assistant-main/unsloth.Q4_K_M.gguf"

def test_model():
    print(f"🔄 모델 로딩 시작: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 에러: 파일이 없습니다 -> {MODEL_PATH}")
        return

    try:
        # 2. 모델 초기화 (디버깅을 위해 verbose=True 설정)
        # n_ctx=8192: 토큰 에러 방지를 위해 컨텍스트 길이를 넉넉하게 잡습니다.
        # n_gpu_layers=-1: 모든 레이어를 GPU에 올립니다 (L4 GPU 필수)
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, 
            n_ctx=8192,      
            verbose=True      # 터미널에 GPU 로드 로그가 찍힙니다.
        )
        print("✅ 모델 로딩 완료!")

        # 3. 테스트 질문 생성
        # Qwen 2.5 Instruct 모델은 ChatML 포맷을 사용합니다.
        messages = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": "B2G 입찰 제안서(RFP)를 작성할 때 가장 중요한 3가지는 뭐야?"}
        ]

        print("\n💬 질문 입력 중...")
        print(f"Q: {messages[1]['content']}\n")

        # 4. 추론 시작 (시간 측정)
        start_time = time.time()
        
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,       # 답변 길이 제한
            temperature=0.3,      # 창의성 조절
            stop=["<|im_end|>", "<|endoftext|>"], # [중요] 이게 없으면 무한 로딩 걸림
            stream=True           # 한 글자씩 출력 테스트
        )

        print("🤖 답변 생성 중:", end=" ", flush=True)
        
        full_response = ""
        for chunk in output:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                token = delta['content']
                print(token, end="", flush=True)
                full_response += token
        
        end_time = time.time()
        print(f"\n\n⏱️ 소요 시간: {end_time - start_time:.2f}초")
        print("✅ 테스트 성공!")

    except Exception as e:
        print(f"\n❌ 테스트 중 에러 발생: {e}")
        # 토큰 에러라면 보통 여기서 ValueError가 뜹니다.

if __name__ == "__main__":
    test_model()