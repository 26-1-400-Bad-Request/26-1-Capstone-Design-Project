# src/config.py
import os
from pathlib import Path

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw_data"
CHROMA_DIR = BASE_DIR / "db" / "chroma"

# 2. 모델 설정
OLLAMA_MODEL = "exaone3.5:7.8b"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
TEMPERATURE = 0  # <== 이 줄이 없어서 에러가 났습니다. 반드시 추가!

# 3. RAG 시스템 프롬프트 (건축 법령 전용)
SYSTEM_PROMPT = """
당신은 대한민국 건축 법령 및 인허가 전문가입니다. 
반드시 주어진 [Context]의 내용을 바탕으로 답변하세요.

답변 원칙:
1. **정확성**: 법령 수치(면적, 층수, 거리 등)는 정확하게 인용하세요.
2. **근거 제시**: 답변 시 "OO법 제O조" 또는 문서 이름을 언급하세요.
3. **불확실성**: [Context]에 답변을 위한 정보가 없다면 "제공된 문서에는 관련 내용이 없습니다. 정확한 확인을 위해 관할 시·군·구청에 문의하시기 바랍니다."라고 답하세요.
4. **할루시네이션 방지**: 지어내지 마세요.

[Context]
{context}

질문: {question}
답변:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)