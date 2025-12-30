# 전처리 스크립트

import os
import re

# 경로는 게인에 맞게 수정
BASE_DIR = r"C:\Users\USER\Desktop\AI_PROJECT\data"
MD_DIR = os.path.join(BASE_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed_data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clean_rag_content(text):
    if not text: return ""
    circled_map = {
        '①': '1. ', '②': '2. ', '③': '3. ', '④': '4. ', '⑤': '5. ',
        '⑥': '6. ', '⑦': '7. ', '⑧': '8. ', '⑨': '9. ', '⑩': '10. ',
        '⑪': '11. ', '⑫': '12. ', '⑬': '13. ', '⑭': '14. ', '⑮': '15. '
    }
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if '|' in line:
            for circled, normal in circled_map.items():
                line = line.replace(circled, normal)
            cleaned_lines.append(line)
            continue
        for circled, normal in circled_map.items():
            line = line.replace(circled, normal)
        line = re.sub(r'\$\\[a-zA-Z]+\s*(\{[^}]*\}|)', ' ', line)
        line = re.sub(r'\$[^\$]+\$', ' ', line)
        line = line.replace('$', '')
        line = line.replace('', '')
        line = re.sub(r'[•ㅇ◼❏◦❍○※■□◆●◎▷▶△▲▽▼]', ' ', line)
        line = re.sub(r' +', ' ', line).strip()
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

print(f"대상 경로: {MD_DIR}")

processed_count = 0

for root, dirs, files in os.walk(MD_DIR):
    for file_name in files:
        if file_name.endswith('.md'):

            # 하위 폴더를 포함한 전체 경로 생성
            input_path = os.path.join(root, file_name)
            output_path = os.path.join(OUTPUT_DIR, file_name)

            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                cleaned_content = clean_rag_content(content)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                processed_count += 1
                print(f"[{processed_count}] 처리 완료: {file_name}")

            except Exception as e:
                print(f"❌ 오류 발생 ({file_name}): {e}")

print(f"\n총 {processed_count}개의 파일 처리가 완료되었습니다.")