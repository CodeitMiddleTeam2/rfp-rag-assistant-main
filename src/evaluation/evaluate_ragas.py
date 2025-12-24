#==================================================================
# 프로그램명: evaluate_ragas.py
# 폴더 위치    : src/evaluation/evaluate_ragas.py
# 프로그램 설명: RAGAS를 활용한 RAG 평가 자동화 스크립트
#             - input: src/dataset/resultdataset_1.json
#             - output: src/dataset/result_YYYYMMDD/ragas_result.csv
# 작성이력 :       
#                 2025.12.18 오민경 최초작성
#==================================================================

#==================================================================
# 프로그램명: evaluate_ragas.py
# 설명: RAGAS를 활용한 RAG 평가 자동화 스크립트
#==================================================================

import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

# --------------------------------------------------
# 1. 프로젝트 루트 기준 경로 설정
# --------------------------------------------------
BASE_DIR = Path("/home/spai0534/rfp-rag-assistant-main")
ENV_PATH = BASE_DIR / ".env"
DATASET_PATH = BASE_DIR / "src" / "dataset" / "ragas_inputs_smk_3.json"

# 결과 저장 경로 (날짜별)
TODAY = datetime.now().strftime("%Y%m%d")
RESULT_DIR = BASE_DIR / "src" / "dataset" / f"result_{TODAY}"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV_PATH = RESULT_DIR / "ragas_result.csv"

# --------------------------------------------------
# 2. 환경 변수 로드
# --------------------------------------------------
load_dotenv(dotenv_path=ENV_PATH)

# --------------------------------------------------
# 3. RAGAS / LLM import
# --------------------------------------------------
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy
)
from langchain_openai import ChatOpenAI

# --------------------------------------------------
# 4. LLM Judge 설정
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# --------------------------------------------------
# 5. 평가 데이터 로드
# --------------------------------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# RAGAS Dataset 생성
ragas_dataset = Dataset.from_list([
    {
        "question": item["question"],
        "contexts": item["contexts"],          # list[str]
        "answer": item["answer"],
        "ground_truth": item["ground_truth"]   # str
    }
    for item in raw_data
])

# --------------------------------------------------
# 6. RAGAS 평가 실행
# --------------------------------------------------
results = evaluate(
    dataset=ragas_dataset,
    metrics=[
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy
    ],
    llm=llm
)

# --------------------------------------------------
# 7. 결과 DataFrame 정리
# --------------------------------------------------
metrics_df = results.to_pandas()

# JSON의 id를 그대로 사용
metrics_df.insert(
    0,
    "id",
    [item["id"] for item in raw_data]
)

# --------------------------------------------------
# 8. CSV 저장
# --------------------------------------------------
metrics_df.to_csv(
    RESULT_CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 9. 출력
# --------------------------------------------------
print("\n===== RAGAS Evaluation Results (Raw) =====")
print(results)

print("\n===== RAGAS Metrics DataFrame =====")
print(metrics_df)

print(f"\n✅ CSV 저장 완료: {RESULT_CSV_PATH}")
