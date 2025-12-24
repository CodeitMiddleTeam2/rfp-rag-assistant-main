import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

#==============================================
# 프로그램명: train_rfp.py
# 폴더위치: src/post_train/train_rfp.py
# 프로그램 설명: unsloth 허브에서 base 모델을 로드하여 사전 학습 시키는 프로그램
# 작성이력: 25.12.22 한상준 최초 작성
#===============================================

model_name = "unsloth/Qwen3-8B" 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 학습 설정 (QLoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# [3. 데이터 포맷팅 함수]
# 사용자님의 JSONL 구조(question, contexts, answer)에 맞춰 프롬프트를 구성합니다.
alpaca_prompt = """당신은 B2G 입찰 전문가입니다. 아래 제공된 컨텍스트를 바탕으로 질문에 정확히 답하세요.

### 질문:
{}

### 참고 컨텍스트:
{}

### 답변:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["question"]
    # contexts 리스트 내의 각 항목을 줄바꿈으로 합쳐 문자열로 만듭니다.
    contexts     = ["\n".join(c) for c in examples["contexts"]] 
    outputs      = examples["answer"]
    texts = []
    for instruction, context, output in zip(instructions, contexts, outputs):
        text = alpaca_prompt.format(instruction, context, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# [4. 데이터셋 로드 및 매핑]
# train_sft.jsonl 파일을 불러와 포맷팅을 적용합니다.
dataset = load_dataset("json", data_files="src/post_train/train_sft.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# [5. 학습 파라미터 조정 (450개 데이터 최적화)]
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # 총 배치 사이즈 = 8
        
        # 데이터가 450개이므로 3번 반복(3 Epochs) 학습을 추천합니다.
        # 약 170~180 Steps 정도 학습이 진행됩니다.
        num_train_epochs = 3, 
        
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        save_strategy = "no",
    ),
)

# [6. 실행 및 저장]
print("🚀 총 450개의 데이터로 RFP 특화 학습을 시작합니다...")
trainer.train()

# GGUF로 즉시 변환 및 저장
model.save_pretrained_gguf("final_model_gguf", tokenizer, quantization_method = "q4_k_m")