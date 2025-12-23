#==============================================
# 프로그램명: embedding_setting.py
# 폴더위치: src/config/embedding_setting.py
# 프로그램 설명: 여기만 고치면 임베딩 모델이 바뀌게 만들기 위한 설정 파일
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

import torch
# 어떤 임베딩 모델 쓸지 "키"로 선택 (아래 레지스트리 키 중 하나)
# - "bge_m3" / "KoE5" / "KURE" / "openai_3_small" / "openai_3_large"
"""
Hugging Face
- bge-m3    : 다국어/범용 성향 강하고 RAG에서 많이 쓰는 편(강한 베이스라인)
- KoE5      : 한국어 검색/리트리벌 쪽으로 유명(한국어 문서면 경쟁력)
- KURE-v1   : KoE5 라인업 쪽이고 bge-m3 기반 파인튜닝으로 소개됨
OpenAI
- open_3_small / large: 최신 임베딩 모델 라인(large가 더 고성능)
"""
EMBEDDING_KEY = "openai_3_small"

# HuggingFace 임베딩에서 cosine 유사도 기반 검색을 쓸 때 보통 정규화 권장
NORMALIZE_EMBEDDINGS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 경로(전처리 된 json 경로 넣어줘야함, json이 리스트여야함)
JSON_PATH = "src/dataset/faiss_metadata.json"
GOLDEN_JSON_PATH = "src/dataset/goldendataset.json"
MAX_DOCS = 2000
# FAISS 검색 파라미터
TOP_K = 5

# 블록타입(header/table/text)을 content 앞에 태그로 붙일지 여부
ADD_TYPE_PREFIX = True

