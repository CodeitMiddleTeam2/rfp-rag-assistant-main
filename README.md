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
│    └── 📂 dataset/            # RAGAS 평가를 위한 dataset저장
│    ├── 📂 evaluation          # RAGAS 평가를 위한 질의응답 배치처리 및 평가
│    ├── 📂 generation          # 답변생성
│    ├── 📂 post_train          # 파인튜닝닝
│    ├── 📂 processing          # 전처리 함수 정의(박지원, 서민경)
│    ├── 📂 vectorestore        # VectorDB 함수 정의(박지원, 서민경)
│    ├── 📂 retrieval/          # RAG검색(한상준, 정예진)
│    ├── 📂 generation/         # 답변생성(한상준, 정예진)

├── .env                        # (비공개) API Key 환경 변수 등(개별)
├── requirements.txt            # 의존성 라이브러리 목록(공통)
├── requirements_jiwon.txt            # 의존성 라이브러리 목록(박지원-전처리 및 임베딩)
├── requirements_yejin.txt            # 의존성 라이브러리 목록(정예진-임베딩모델비교 및 retrieve)
├── requirements_sangjun.txt            # 의존성 라이브러리 목록(한상준-모델양자화/파인튜닝, RAG-Chain)
├── requirements_minkyungoh.txt            # 의존성 라이브러리 목록(오민경-goldendataset, RAGAS)
└── README.md                   # 프로젝트 문서(공통)

```
## 파일리스트
|폴더|파일명|주요내용|담당|비고|
|-|-|-|-|-|
| data|local파일|개별로 hwp/pdf/전처리 결과 파일 저장|공통|사이즈가 큰 관계로 local에서 관리|
| notebook|eda_example.ipynb|csv메타데이터 확인|오민경|-|
| notebook|ragas_result.ipynb|rags결과 csv로 저장한 것을 확인 및 시각화|오민경|-|
| metadata|create_table.sql|supabase table 및 index 생성 스크립트|박지원/서민경|-|
| metadata|create_function.sql|supbase vector 및 키워드 검색을 위한 function 생성 스크립트|오민경|-|
| src/dataset|goldendataset.json|테스트용 질문/답변 set|오민경|-|
| src/dataset|openai_result.json|LLM openai모델 적용 결과 context & 답변|오민경|-|
| src/dataset|qwen_result.json|LLM qwen모델 적용 결과 context & 답변|오민경|-|
| src/dataset|ragas_input.json|LLM open모델 및 context수정 적용 결과 context & 답변|오민경|-|
| src/evaluation|evaluate_goldendataset_XXX.py|goldendataset을 가지고 context/답변 생성 파이프라인|오민경|-|
| src/evaluation|evaluate_ragas.py|evaluate_goldendataset_XXX.py수행결과 파일을 가지고 RAGASE평가수행|오민경|-|
| src/generation|app.py|RAG 기반 RFP 분석 플랫폼 (DB + Rerank + Local LLM)|한상준|-|
| src/generation|load_local_model.py|학습된 로컬 모델을 불러오는 모듈 함수|한상준|-|
| src/generation|model_manager.py|웹 데모에서 모델을 선택하게끔(로컬 or API) 만들어주는 매니저 클래스|한상준|-|
| src/generation|supabase_manager.py|supabase DB 를 웹 데모에 연동하기 위한 클래스|한상준|-|
| src/generation|test_local_model.py|학습된 로컬 모델을 실험해보기 위한 파일|한상준|-|
| src/post_train|aumented_dataset.json|학습데이터 증강|한상준|-|
| src/post_train|augmented_train_data.py|원본 질문-답 데이터를 증강시켜 학습 데이터셋을 생성하는 파일|한상준|-|
| src/post_train|convert_gguf.py|사전학습 시킨 모델을 gguf 파일로 변환하는 프로그램|한상준|-|
| src/post_train|merge_and_convert.py|학습 데이터셋들을 모아서 병합하고 학습 규격으로 변환하는 프로그램|한상준|-|
| src/post_train|train_rfp.py|unsloth 허브에서 base 모델을 로드하여 사전 학습 시키는 프로그램|한상준|-|
| src/post_train|train_sft.sonl|sft를 위한 sonl파일|한상준|-|
| src/processing|build_rag_pjw.py|json으로 파일 만들기|박지원|-|
| src/processing|hwp_to_pdf_pjw|한글파일 pdf로 변환|박지원|-|
| src/processing|preprocess_pjw.py|전처리|박지원|-|
| src/processing|upload_chunks_final|청크 supabase에 업로드||-|
| src/processing|vision_process_pwj|vlm처리|박지원|-|





## 협업일지 링크
- 한상준 https://drive.google.com/file/d/1IXDBrduZ9yFhgFZW-hwiaJo4hvRx777c/view?usp=sharing
- 박지원 https://www.notion.so/2c602918343a80bdbc0ada371a76dca7
- 서민경 
- 정예진 https://www.notion.so/2a0fce412ebd8001be51dfdefe7fce90?v=2a0fce412ebd80ad9c11000cf704c795&source=copy_link
- 오민경 https://www.notion.so/2c5208dedd488008a97bed963a21fc86

## 작성이력
- 최  초 작성일: 2025.12.14
- 마지막 수정일: 2026.01.02
