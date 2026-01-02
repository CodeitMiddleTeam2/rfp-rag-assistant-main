# 📂 RFP-RAG-Assistant

> **사내 RFP 문서 분석 및 질의응답을 위한 RAG 기반 지식 관리 시스템**

이 프로젝트는 100여 개의 실제 **제안 요청서(RFP)**와 메타데이터를 기반으로 구축된 **RAG(Retrieval-Augmented Generation)** 시스템 입니다. 사용자의 자연어 질문에 대해 관련 RFP 문서를 검색하고, 핵심 내용을 요약 및 추출하여 답변을 제공합니다.

## 🎯 프로젝트 목적

- **효율적인 정보 검색:** 방대한 RFP 문서 아카이브에서 필요한 정보를 빠르게 검색.
- **문맥 기반 Q&A:** 단순 키워드 매칭이 아닌, 문서의 내용을 이해하고 질문에 답변.
- **자동 요약:** 긴 제안 요청서의 핵심 요구사항과 과업 내용을 요약하여 제공.
- **메타데이터 활용:** CSV 메타데이터와 연동하여 발주처, 기간 등 필터링 검색 구현.

## 🛠️ 주요 기능

1. **문서 수집 및 전처리 (Ingestion):** PDF 텍스트 추출 및 의미 단위 청킹(Chunking).
2. **임베딩 및 저장 (Vector Store):** 텍스트를 벡터화하여 의미 기반 검색이 가능하도록 DB에 저장.
3. **하이브리드 검색 (Retrieval):** 의미(Semantic) 검색과 메타데이터 필터링을 결합.
4. **답변 생성 (Generation):** 검색된 문맥(Context)을 바탕으로 LLM이 정확한 답변 생성.

## 🏗️ 기술 스택 (Tech Stack)

- **Language:** Python 3.12+
- **LLM Orchestration:** LangChain
- **LLM Model:** OpenAI GPT-5 (예정)
- **Vector DB:** ChromaDB (or FAISS)
- **Embedding:** OpenAI Embeddings / HuggingFace (예정)
- **Reranker:** BGE(예정)
- **Data Processing:** Pandas, PyPDFLoader


## 📂 폴더 구조

```
rfp-rag-assistant-main/
├── 📂 data/                    # (비공개) RFP 원본 및 전처리 데이터(담당: 개별)
├── 📂 notebooks/               # 데이터 탐색(EDA) 및 모델 실험용 노트북(담당: 개별)
├── 📂 metadata/                # Vector DB 메타데이터 구조 정의(담당: 박지원, 서민경)
├── 📂 src/                     # RAG 시스템 핵심 소스 코드
│    └── main.py                
│    └── config.py               # 설정 파라미터 (오민경)
│    └── 📂 data/               # Vector DB(박지원, 서민경)
│    │    └── chroma_db          
│    ├── 📂 processing          # 전처리 함수 정의(박지원, 서민경)
│    ├── 📂 vectorestore        # VectorDB 함수 정의(박지원, 서민경)
│    ├── 📂 retrieval/          # RAG검색(한상준, 정예진)
│    ├── 📂 generation/         # 답변생성(한상준, 정예진)
├── 📂 prompt/                  # prompt template (공통)
├── 📂 pipelines                # 함수 단계별 순차처리 (공통)
├── .env                        # (비공개) API Key 환경 변수 등(개별)
├── requirements.txt            # 의존성 라이브러리 목록(공통)
├── requirements_jiwon.txt            # 의존성 라이브러리 목록(박지원-전처리 및 임베딩)
├── requirements_yejin.txt            # 의존성 라이브러리 목록(정예진-임베딩모델비교 및 retrieve)
├── requirements_sangjun.txt            # 의존성 라이브러리 목록(한상준-모델양자화/파인튜닝, RAG-Chain)
├── requirements_minkyungoh.txt            # 의존성 라이브러리 목록(오민경-goldendataset, RAGAS)
└── README.md                   # 프로젝트 문서(공통)

```
## 최  초 작성일: 2025.12.14
## 마지막 수정일: 2026.01.02
