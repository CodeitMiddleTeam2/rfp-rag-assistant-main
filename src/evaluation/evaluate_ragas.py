#==================================================================
# 프로그램명: evaluate_ragas.py
# 폴더 위치    : src/evaluation/evaluate_ragas.py
# 프로그램 설명: RAGAS를 활용한 RAG 평가 자동화 스크립트
#             - input: src/dataset/ragas_inputs_xxx_x.json
#             - output: src/dataset/result_YYYYMMDD/ragas_result.csv
# 작성이력 :       
#                 2025.12.18 오민경 최초작성
#==================================================================

#==================================================================
# 프로그램명: evaluate_ragas.py
# 설명: RAGAS를 활용한 RAG 평가 자동화 스크립트 (안정판)
#==================================================================

import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
import re

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
BASE_DIR = Path("/home/spai0534/rfp-rag-assistant-main")
ENV_PATH = BASE_DIR / ".env"
DATASET_PATH = BASE_DIR / "src" / "dataset" / "openai_result.json"

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
# 4. LLM Judge
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# --------------------------------------------------
# 5. 안전한 JSON 로더
# --------------------------------------------------
def load_dirty_json(path: Path):
    """
    id 필드를 기준으로 JSON 객체를 분리하여 복구
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # 위험 문자 제거
    raw = raw.replace("\uf000", "").replace("\u0000", "")

    # id 기준으로 객체 분리
    chunks = re.split(r'(?=\{\s*"id"\s*:\s*")', raw)

    objects = []
    failed = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("{"):
            continue

        # 끝에 , 제거
        chunk = re.sub(r",\s*$", "", chunk)

        try:
            obj = json.loads(chunk)
            objects.append(obj)
        except json.JSONDecodeError:
            failed.append(chunk[:200])  # 앞부분만 기록

    print(f"✅ 복구 성공: {len(objects)}개")
    print(f"❌ 복구 실패: {len(failed)}개")

    if not objects:
        raise RuntimeError("유효한 JSON 객체를 하나도 복구하지 못했습니다.")

    return objects

# --------------------------------------------------
# 6. 정제 함수
# --------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\uf000", "")
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)  # 단독 숫자 제거
    return text.strip()

def clean_contexts(contexts):
    cleaned = []
    for c in contexts:
        if isinstance(c, str):
            c = c.replace("\uf000", "").strip()
            if c:
                cleaned.append(c)
    return cleaned

# --------------------------------------------------
# 7. 데이터 로드 + 정제
# --------------------------------------------------
raw_data = load_dirty_json(DATASET_PATH)

processed_data = []
skipped_ids = []

for item in raw_data:
    contexts = clean_contexts(item.get("contexts", []))

    if not contexts:
        skipped_ids.append(item.get("id"))
        continue

    processed_data.append({
        "id": item.get("id"),
        "question": clean_text(item.get("question")),
        "contexts": contexts,
        "answer": clean_text(item.get("answer")),
        "ground_truth": clean_text(item.get("ground_truth")),
    })

print(f"▶ 평가 대상 샘플 수: {len(processed_data)}")
print(f"▶ 제외된 샘플 ID: {skipped_ids}")

# --------------------------------------------------
# 8. RAGAS Dataset 생성
# --------------------------------------------------
ragas_dataset = Dataset.from_list([
    {
        "question": d["question"],
        "contexts": d["contexts"],
        "answer": d["answer"],
        "ground_truth": d["ground_truth"],
    }
    for d in processed_data
])

# --------------------------------------------------
# 9. RAGAS 평가
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
# 10. 결과 정리
# --------------------------------------------------
metrics_df = results.to_pandas()

metrics_df.insert(0, "id", [d["id"] for d in processed_data])
metrics_df.insert(1, "question", [d["question"] for d in processed_data])

metrics_df.to_csv(
    RESULT_CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 11. 출력
# --------------------------------------------------
print("\n===== RAGAS Evaluation Results =====")
print(results)

print("\n===== RAGAS Metrics DataFrame =====")
print(metrics_df)

print(f"\n✅ CSV 저장 완료: {RESULT_CSV_PATH}")
