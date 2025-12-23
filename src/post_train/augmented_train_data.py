import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# [환경 변수 로드]
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("🚨 API Key가 없습니다. .env 파일의 OPENAI_API_KEY를 확인해주세요!")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)

def augment_data(output_filename, augmentation_factor=5):
    """
    작성 의도: Golden Dataset의 문맥을 유지하며 LLM을 통해 학습 데이터를 증강함
    작용: 50개의 샘플을 시드로 하여 설정된 배수만큼 데이터 양을 늘림
    """
    
    # 1. 경로 설정 (스크립트 위치 기준)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, '..', 'dataset', 'goldendataset.json')
    output_file = os.path.join(current_dir, output_filename)

    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    augmented_results = []

    for item in original_data:
        print(f"🔄 증강 중: {item['id']} ({item['metadata'].get('dataset_type', 'N/A')})")
        
        # 2. GPT에게 전달할 프롬프트 설계 (JSON 키값 명시)
        prompt = f"""
        당신은 B2G 입찰 컨설팅 데이터 엔지니어입니다. 
        아래 제공된 '원본 데이터'의 [컨텍스트]를 바탕으로, 새로운 [질문]과 [답변] 쌍을 {augmentation_factor}개 생성하세요.
        
        [원본 컨텍스트]:
        {item['contexts']}
        
        [가이드라인]:
        1. 질문의 형식을 다양하게 하세요 (요약 요청, 특정 수치 추출, 비교 분석 등).
        2. 답변은 반드시 제공된 컨텍스트에 근거해야 합니다.
        3. 'samples'라는 키를 가진 JSON 객체 내에 리스트 형식으로 출력하세요.
        4. 형식: {{"samples": [{{"question": "...", "answer": "...", "ground_truth": "..."}}]}}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )

            # 3. 생성된 데이터 파싱
            response_data = json.loads(response.choices[0].message.content)
            new_samples = response_data.get("samples", [])

            # 4. ID 부여 및 메타데이터 상속 (학습 시 데이터 추적을 위해 필수)
            for i, sample in enumerate(new_samples):
                sample["id"] = f"aug_{item['id']}_{i+1}"
                sample["contexts"] = item["contexts"] # 원본 컨텍스트 상속
                sample["metadata"] = item["metadata"] # 메타데이터 상속
                augmented_results.append(sample)
                
        except Exception as e:
            print(f"⚠️ {item['id']} 증강 중 에러 발생: {e}")

    # 5. 최종 결과 저장 (JSONL 대신 JSON으로 먼저 저장하여 가독성 확보)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 증강 완료! 총 {len(augmented_results)}개의 데이터가 {output_file}에 저장되었습니다.")

# 실행 (증강 배수는 테스트를 위해 3 정도로 시작하는 것을 추천합니다)
if __name__ == "__main__":
    augment_data('augmented_dataset2.json', augmentation_factor=6)