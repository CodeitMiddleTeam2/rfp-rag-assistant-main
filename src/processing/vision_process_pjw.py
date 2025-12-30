# 이미지를 VLM으로 요약하여 내용 추출

import base64
from openai import OpenAI
import re
from pathlib import Path
import sys
import shutil 


# API 키 설정 (보안 문제로 API key는 생략)
client = OpenAI(api_key="sk-...")

# 제외할 폴더 이름 (이 이름이 경로에 포함되면 건너뜀)
EXCLUDE_FOLDER = "original_backup"


IMAGE_PATTERN = re.compile(r'!\[.*?\]\(([^)]+)\)')

def image_to_text(image_path: Path) -> str:
    """이미지를 GPT-5-mini로 분석하여 텍스트 추출"""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "이 이미지는 문서에 포함된 다이어그램 또는 도표입니다.\n"
                                "이미지에 포함된 모든 텍스트를 최대한 정확히 추출하고,\n"
                                "흐름이나 구조가 있다면 문단 형태로 정리해 주세요.\n"
                                "한글 위주로 출력해 주세요."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API 호출 오류: {e}"

def process_file(src_path: Path, output_root: Path, src_root: Path):
    """단일 파일을 처리하여 결과 폴더에 저장"""
    try:
        content = src_path.read_text(encoding="utf-8")
        md_dir = src_path.parent

        def replace_image(match):
            img_rel_path = match.group(1)
            import urllib.parse
            img_rel_path = urllib.parse.unquote(img_rel_path)
            
            img_path = (md_dir / img_rel_path).resolve()

            if not img_path.exists():
                return match.group(0)

            print(f"[이미지 분석 중]: {img_path.name}")
            
            extracted_text = image_to_text(img_path)

            return (
                match.group(0)
                + "\n\n> **[이미지 내용 설명]**\n"
                + "\n".join(f"> {line}" for line in extracted_text.splitlines())
                + "\n"
            )

        new_content = IMAGE_PATTERN.sub(replace_image, content)

        # 저장 경로 계산
        rel_path = src_path.relative_to(src_root)
        dest_path = output_root / rel_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(new_content, encoding="utf-8")
        print(f" ✅ 저장 완료: {dest_path.name}")

    except Exception as e:
        print(f" ❌ 실패 ({src_path.name}): {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python vision_process.py <대상_폴더_경로>")
        sys.exit(1)

    target_dir = Path(sys.argv[1]).resolve()

    if not target_dir.exists():
        print(f"오류: 폴더가 없습니다: {target_dir}")
        sys.exit(1)

    # 결과 폴더 생성
    output_dir = target_dir.parent / f"{target_dir.name}_vision_LLM"
    output_dir.mkdir(exist_ok=True)

    print(f"[작업 시작]")
    print(f"📂 대상 폴더: {target_dir}")
    print(f"📂 결과 폴더: {output_dir}")
    print(f"[제외할 폴더명]: '{EXCLUDE_FOLDER}'\n")

    # 모든 md 파일 찾기
    all_files = list(target_dir.rglob("*.md"))
    
    # 백업 폴더 필터링
    # 파일 경로 중에 'original_backup'이라는 단어가 포함되어 있으면 리스트에서 뺍니다.
    target_files = [f for f in all_files if EXCLUDE_FOLDER not in str(f)]

    print(f"총 {len(all_files)}개 파일 중 백업 폴더를 제외하고 {len(target_files)}개를 처리합니다.")

    for i, md_file in enumerate(target_files, 1):
        print(f"\n[{i}/{len(target_files)}] 처리 중: {md_file.name}")
        process_file(md_file, output_dir, target_dir)

    print(f"\n[완료] - '{output_dir}' 폴더를 확인하세요.")