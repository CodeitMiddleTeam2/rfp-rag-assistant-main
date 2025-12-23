#==============================================
# 프로그램명: retriever_setting.py
# 폴더위치: src/config/retriever_setting.py
# 프로그램 설명: Retriever(검색 방식)만 따로 관리하는 설정 파일
# (LangChain RAG에서 검색 전략을 쉽게 바꿔가며 비교하려고 만들었음)
# 작성이력: 2025.12.18 정예진 최초 작성
#===============================================

"""
검색 방식 선택 (Retriever Key)
- dense         : 벡터 검색만 (FAISS/VectorDB)
- bm25          : 키워드(BM25) 검색만
- hybrid_rrf    : BM25 + Dense를 RRF(EnsembleRetriever)로 결합
"""
RETRIEVER_KEY = "dense"

# 검색 단계에서 뽑아오는 문서 수
TOP_K = 8

# =============================================
# 하이브리드 결합 가중치 (hybrid_rrf에서 사용)
# =============================================
# retrievers=[bm25, dense] 기준 weights
# bm25가 강하면 키워드 정확도 올라감 / dense가 강하면 표현 다양성/의미 매칭 올라감
HYBRID_WEIGHTS = [0.5,0.5]

# =====================================
# Reranking 설정 (선택)
# =====================================
# rerank를 따로 비교하고 싶으면 아래 사용
"""
- none              : 없음
- bge_rerank_m3     : BAAI/bge-reranker-v2-m3 (멀티링구얼)
- bge_rerank_m3_ko  : 한국어 최적화 커뮤니티 버전(있으면)
- ko_reranker
- jina_reranker_v2
"""
RERANKER_KEY = "bge_rerank_m3"

# rerank 후 최종으로 남길 문서 수(보통 TOP_K보다 작게)
RERANK_TOP_N = 5
# rerank할 후보 문서 수 (TOP_K보다 크게 잡는 게 보통)
RERANK_CANDIDATE_K = 30

# 백엔드 선택 키
RETRIEVAL_BACKEND = "local"     # supabase