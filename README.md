# 건축 법령 RAG 보좌관

대한민국 건축·인허가 관련 문서(PDF, YAML)를 검색하고, **로컬 LLM(Ollama)** 으로 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다. Streamlit 웹 UI에서 질문을 입력하면 법령 맥락을 바탕으로 답변과 참조 문서를 확인할 수 있습니다.

## 주요 기능

- **문서 기반 답변**: Chroma 벡터 DB에서 관련 조문·규정을 검색한 뒤 LLM이 답변 생성
- **할루시네이션 완화**: 제공된 Context에 없는 내용은 추측하지 않고, 관할 기관 문의 등으로 안내
- **근거 확인**: UI에서 검색된 문서 청크(파일명·본문 일부) 확인
- **로컬 실행**: Ollama로 LLM을 PC에서 실행 (API 키 불필요)

## 기술 스택

| 구분 | 기술 |
|------|------|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Ollama](https://ollama.com/) (`ChatOllama`) |
| 임베딩 | `jhgan/ko-sroberta-multitask` (HuggingFace) |
| 벡터 DB | [Chroma](https://www.trychroma.com/) |
| 프레임워크 | LangChain |

## 프로젝트 구조

```
26-1-Capstone-Design-Project/
├── data/
│   ├── raw_data/          # PDF 원문 (직접 추가)
│   └── rules_yaml/        # 구조화된 규정 YAML
├── db/
│   └── chroma/            # 벡터 DB (build_vector_db 실행 후 생성·갱신)
├── src/
│   ├── main.py            # Streamlit 앱 진입점
│   ├── rag.py             # RAG 체인·리트리버
│   ├── config.py          # 경로, 모델, 시스템 프롬프트
│   └── build_vector_db.py # PDF/YAML → Chroma 인덱싱
├── requirements.txt
└── README.md
```

## 사전 요구 사항

- **Python** 3.10 이상 권장
- **Ollama** 설치 및 실행 ([다운로드](https://ollama.com/download))
- 기본 LLM 모델: `exaone3.5:7.8b` (다른 모델 사용 시 UI 또는 `src/config.py`에서 변경 가능)
- (선택) GPU가 있으면 임베딩·추론 속도 향상

## 설치

프로젝트 루트에서 가상환경을 만들고 의존성을 설치합니다.

```powershell
cd 26-1-Capstone-Design-Project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install streamlit langchain-ollama langchain-core langchain-text-splitters sentence-transformers torch
```

> `requirements.txt`에는 LangChain·Chroma·PDF 처리 등 핵심 패키지만 포함되어 있습니다. 웹 UI와 Ollama 연동을 위해 위 추가 패키지 설치가 필요합니다.

### Ollama 모델 준비

별도 터미널에서 Ollama를 실행한 뒤, 사용할 모델을 받습니다.

```powershell
ollama serve
ollama pull exaone3.5:7.8b
```

## 벡터 DB 구축

문서를 추가·수정했거나 `db/chroma`가 없을 때, 아래 명령으로 인덱스를 만듭니다.

```powershell
python -m src.build_vector_db
```

**데이터 소스**

| 경로 | 설명 |
|------|------|
| `data/raw_data/*.pdf` | 건축·인허가 관련 PDF |
| `data/rules_yaml/*.yaml` | 용도지역, 면적·층수, 설비 등 규정 YAML |

처리 흐름: 문서 로드 → 법령 구조(장·조·항 등) 기준 청킹 → 한국어 임베딩 → `db/chroma`에 저장.

## 실행 방법

`main.py`는 **Streamlit 앱**이므로 `python src/main.py`가 아니라 아래처럼 실행합니다.

```powershell
streamlit run src/main.py
```

브라우저에서 **http://localhost:8501** 이 열립니다.

1. 하단 입력창에 건축·인허가 질문 입력
2. AI 답변 확인
3. **「근거 문서 확인」** 펼쳐서 참조한 문서 청크 확인

**실행 전 체크리스트**

- [ ] 가상환경 활성화 및 패키지 설치 완료
- [ ] `db/chroma` 존재 (없으면 `python -m src.build_vector_db` 실행)
- [ ] Ollama 실행 중 (`http://localhost:11434`)
- [ ] 사용 모델이 `ollama pull` 로 설치됨

## 설정 변경

`src/config.py`에서 다음을 조정할 수 있습니다.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_MODEL` | `exaone3.5:7.8b` | Ollama 모델명 |
| `EMBEDDING_MODEL_NAME` | `jhgan/ko-sroberta-multitask` | 검색용 임베딩 모델 |
| `TEMPERATURE` | `0` | 답변 무작위성 (0에 가까울수록 일관적) |
| `SYSTEM_PROMPT` | (건축 전문가 프롬프트) | 답변 톤·근거·불확실성 처리 규칙 |

Streamlit 사이드바에서도 **Ollama 모델명**, **참조 문서 개수(k)** 를 실행 중에 바꿀 수 있습니다.

## 동작 개요

```mermaid
flowchart LR
    A[사용자 질문] --> B[Chroma 검색]
    B --> C[관련 문서 청크]
    C --> D[프롬프트 + Context]
    D --> E[Ollama LLM]
    E --> F[답변 + 근거 문서 UI]
```

1. 질문을 임베딩하여 Chroma에서 상위 k개 청크 검색
2. 검색 결과를 `SYSTEM_PROMPT`의 `[Context]`에 삽입
3. Ollama LLM이 Context만 근거로 답변 생성
4. UI에 답변과 검색된 청크 목록 표시

## 문제 해결

| 증상 | 해결 |
|------|------|
| `Ollama가 켜져 있는지 확인` | `ollama serve` 실행, 모델 `ollama pull` 여부 확인 |
| `ModuleNotFoundError: streamlit` | `pip install streamlit` |
| `ModuleNotFoundError: langchain_ollama` | `pip install langchain-ollama` |
| 검색 결과가 비어 있거나 부정확 | `data/` 문서 확인 후 `python -m src.build_vector_db` 재실행 |
| 첫 실행이 매우 느림 | 임베딩 모델(`ko-sroberta-multitask`) 최초 다운로드 — 정상 동작 |

## 유의 사항

- 본 시스템은 **참고용 보조 도구**이며, 최종 인허가·법적 판단은 관할 시·군·구청 등 공식 기관 확인이 필요합니다.
- Context에 없는 질문은 모델이 “문서에 없음”으로 응답하도록 프롬프트가 설정되어 있습니다.

## 라이선스

종합설계(Capstone Design) 프로젝트용 저장소입니다. 세부 라이선스는 팀 정책에 따릅니다.
