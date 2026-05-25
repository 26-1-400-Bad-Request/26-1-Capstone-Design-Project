# src/build_vector_db.py
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. 기본 경로 및 설정 로드 ---
try:
    from src.config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME
except ModuleNotFoundError:
    from config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME

# ⭐️ YAML 파일이 있는 폴더 경로 설정 (DATA_DIR과 같은 위치의 rules_yaml 폴더로 가정)
# 예: C:\...\data\rules_yaml
YAML_DIR = os.path.join(os.path.dirname(DATA_DIR), "rules_yaml")

def build_db():
    documents = []
    
    # --- 2-1. PDF 파일 로드 (raw_data 폴더) ---
    print(f"1. '{DATA_DIR}'에서 PDF 파일들을 불러오는 중입니다...")
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(DATA_DIR, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ {filename} PDF 로드 실패: {e}")
    else:
        print(f"⚠️ 에러: {DATA_DIR} 폴더를 찾을 수 없습니다.")

    # --- 2-2. YAML 파일 로드 (rules_yaml 폴더) ---
    print(f"2. '{YAML_DIR}'에서 YAML 파일들을 불러오는 중입니다...")
    if os.path.exists(YAML_DIR):
        for filename in os.listdir(YAML_DIR):
            if filename.endswith((".yaml", ".yml")):
                file_path = os.path.join(YAML_DIR, filename)
                try:
                    # 인코딩 에러 방지를 위해 utf-8 명시
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"⚠️ {filename} YAML 로드 실패: {e}")
    else:
        print(f"⚠️ 에러: {YAML_DIR} 폴더를 찾을 수 없습니다. 경로가 맞는지 확인해주세요.")

    # --- 3. 빈 데이터 필터링 및 청킹 ---
    # 빈 텍스트 페이지 필터링 (스캔본 PDF 등)
    valid_documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"\n총 {len(documents)} 페이지/파일 중 텍스트가 정상 추출된 {len(valid_documents)} 개의 데이터를 불러왔습니다.")
    
    if len(valid_documents) == 0:
        print("❌ 추출된 텍스트가 없습니다. 데이터 폴더와 파일 상태를 확인해주세요.")
        return

    print("3. 법령 구조(조/항/호) 기반으로 텍스트를 청킹하는 중입니다...")

    custom_separators = [
        r"\n제\s*\d+\s*장",       # 1순위: '제 1 장' 등 장 단위
        r"\n제\s*\d+\s*조",       # 2순위: '제 11 조' 등 조 단위
        r"\n\s*[①②③④⑤⑥⑦⑧⑨⑩]", # 3순위: 동그라미 기호 (항 단위)
        r"\n\s*\d+\.",          # 4순위: '1.', '2.' (호 단위)
        r"\n\s*[가-하]\.",       # 5순위: '가.', '나.' (목 단위)
        "\n\n",                 # 6순위: 일반 단락 변경
        "\n",                   # 7순위: 줄바꿈
        r"\. ",                 # 8순위: 문장 끝
        " ",                    # 9순위: 띄어쓰기
        ""                      # 10순위: 글자 단위 
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=custom_separators,
        is_separator_regex=True
    )

    chunks = text_splitter.split_documents(valid_documents)
    print(f"총 {len(chunks)} 개의 텍스트 청크로 분할되었습니다.")

    # --- 4. 벡터 변환 및 ChromaDB 저장 ---
    if len(chunks) > 0:
        print(f"4. '{EMBEDDING_MODEL_NAME}' 임베딩 모델 로드 및 벡터 DB 저장을 시작합니다...")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        print(f"🎉 완료! 법령 맞춤형 벡터 DB가 '{CHROMA_DIR}' 폴더에 성공적으로 저장되었습니다.")
    else:
        print("❌ 텍스트 청크 분할에 실패했습니다.")

if __name__ == "__main__":
    build_db()