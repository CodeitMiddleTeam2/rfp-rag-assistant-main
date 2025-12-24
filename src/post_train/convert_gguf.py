from unsloth import FastLanguageModel

#==============================================
# 프로그램명: convert_gguf.py
# 폴더위치: src/post_train/convert_gguf.py
# 프로그램 설명: 사전학습 시킨 모델을 gguf 파일로 변환하는 프로그램
# 작성이력: 25.12.22 한상준 최초 작성
#===============================================

# 1. 방금 학습을 마치고 저장된 모델 불러오기
# (final_model_gguf 폴더에 16-bit 상태로 저장되어 있습니다)
print("📂 저장된 모델을 로드합니다...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./final_model_gguf", 
    load_in_4bit = False, # 변환을 위해 원본(16bit) 정밀도로 로드
)

# 2. GGUF 변환 다시 시도
print("💾 GGUF 변환을 재시도합니다...")
model.save_pretrained_gguf(
    "final_model_gguf", 
    tokenizer, 
    quantization_method = "q4_k_m"
)
print("✅ 변환 성공!")