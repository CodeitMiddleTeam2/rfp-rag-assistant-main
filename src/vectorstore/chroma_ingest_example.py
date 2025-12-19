import os
import json
import uuid
from datetime import datetime

from processing.embedder import embed_text
from vectorstore.chroma_client import get_chroma_collection
from vectorstore.metadata_loader import load_metadata_map

# -------------------------------------------------
# 경로 설정
# -------------------------------------------------
CHUNK_DIR = r"C:\Users\mungm\Documents\ai_engineer\genai_rag2\data\output\json_chunking"


# -------------------------------------------------
# Chroma metadata 정제 (🔥 매우 중요)
# -------------------------------------------------
def sanitize_metadata(meta: dict) -> dict:
    """
    Chroma metadata 규칙:
    - None ❌
    - list / dict ❌  → JSON 문자열로 변환
    - 허용: str, int, float, bool
    """
    clean = {}

    for k, v in meta.items():
        if v is None:
            continue
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)

    return clean


# -------------------------------------------------
# Chroma Ingest 메인 로직
# -------------------------------------------------
def ingest_chunks_to_chroma(log_every=50):
    collection = get_chroma_collection()
    excel_meta_map = load_metadata_map()

    files = [f for f in os.listdir(CHUNK_DIR) if f.endswith("_chunking.json")]
    total_files = len(files)

    total_chunks = 0
    total_saved = 0

    print("\n🚀 Chroma Ingest 시작")
    print(f"📂 대상 파일 수: {total_files}")

    for file_idx, file in enumerate(files, start=1):
        file_key = file.replace("_chunking.json", "")

        print(f"\n▶ [{file_idx}/{total_files}] 파일 처리 시작")
        print(f"   - 파일명: {file}")

        base_meta = excel_meta_map.get(file_key)
        if not base_meta:
            print("   ⚠ Excel 메타 없음 → skip")
            continue

        with open(os.path.join(CHUNK_DIR, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        file_chunk_total = len(chunks)
        file_saved = 0

        print(f"   - 총 chunk 수: {file_chunk_total}")

        for chunk_idx, ch in enumerate(chunks, start=1):
            total_chunks += 1

            text = ch.get("chunk", "")
            if not isinstance(text, str) or not text.strip():
                continue

            embedding = embed_text(text)

            # chunk JSON 메타 전체 포함 (chunk, chunk_id 제외)
            chunk_meta = {k: v for k, v in ch.items() if k not in ("chunk", "chunk_id")}

            raw_metadata = {
                **base_meta,  # 문서 단위 메타 (Excel)
                **chunk_meta,  # 청크 단위 메타 (page, section 등)
                "ingested_at": datetime.now().isoformat(),
            }

            metadata = sanitize_metadata(raw_metadata)

            collection.add(
                ids=[ch.get("chunk_id", str(uuid.uuid4()))],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
            )

            file_saved += 1
            total_saved += 1

            # 진행 로그
            if chunk_idx % log_every == 0 or chunk_idx == file_chunk_total:
                print(
                    f"     · chunk {chunk_idx}/{file_chunk_total} "
                    f"(파일 저장 {file_saved})"
                )

        print(f"  ✔ 파일 완료: {file_saved}/{file_chunk_total} chunks 저장")

    print("\n🎉 Chroma Ingest 완료")
    print(f"📊 전체 chunk 수: {total_chunks}")
    print(f"💾 저장된 chunk 수: {total_saved}")


# -------------------------------------------------
# 실행부
# -------------------------------------------------
if __name__ == "__main__":
    ingest_chunks_to_chroma(log_every=50)
