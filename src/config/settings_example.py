# settings.py
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "rag_chunks"

DENSE_WEIGHT = 0.7
LEX_WEIGHT = 0.3
FINAL_TOP_K = 5  # 🔹 rerank 후 최종 문서 개수
