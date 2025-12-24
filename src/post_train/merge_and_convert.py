import json
import os

#==============================================
# 프로그램명: merge_and_convert.py
# 폴더위치: src/post_train/merge_and_convert.py
# 프로그램 설명: 학습 데이터셋들을 모아서 병합하고 학습 규격으로 변환하는 프로그램
# 작성이력: 25.12.22 한상준 최초 작성
#===============================================

# 작성 의도: 흩어져 있는 증강 데이터들을 하나로 통합하고 학습 규격(JSONL)으로 변환합니다.
def merge_to_jsonl(file_list, output_filename):
    total_samples = 0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for file_name in file_list:
            input_path = os.path.join(current_dir, file_name)
            
            if not os.path.exists(input_path):
                print(f"⚠️ 파일을 찾을 수 없습니다: {file_name}")
                continue

            with open(input_path, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
                for item in data:
                    # JSON 객체를 한 줄의 문자열로 변환하여 저장
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    total_samples += 1
            
            print(f"📖 {file_name} 처리 완료...")

    print(f"✅ 통합 완료! 총 {total_samples}개의 데이터가 {output_path}에 저장되었습니다.")

# 실행: 두 파일을 합쳐서 train_sft.jsonl 생성
if __name__ == "__main__":
    files_to_merge = ['augmented_dataset.json', 'augmented_dataset2.json']
    merge_to_jsonl(files_to_merge, 'train_sft.jsonl')