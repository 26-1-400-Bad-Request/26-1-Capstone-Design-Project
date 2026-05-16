import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. 기본 경로 설정 ---
pdf_directory = "./data/raw_data"      
persist_directory = "./chroma_db"     

# --- 2. PDF 파일 로드 ---
print("1. PDF 파일들을 불러오는 중입니다...")
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        # PyPDFLoader는 파일명과 페이지 번호를 메타데이터로 자동 추출해 줍니다.
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

print(f"총 {len(documents)} 페이지의 데이터를 불러왔습니다.")

# --- 3. ⭐️ 법령 맞춤형 데이터 청킹 (핵심) ⭐️ ---
print("2. 법령 구조(조/항/호) 기반으로 텍스트를 청킹하는 중입니다...")

# 정규표현식(Regex)을 사용하여 법령의 목차 구조대로 텍스트를 분할합니다.
custom_separators = [
    r"\n제\s*\d+\s*장",       # 1순위: '제 1 장' 등 장 단위
    r"\n제\s*\d+\s*조",       # 2순위: '제 11 조' 등 조 단위
    r"\n\s*[①②③④⑤⑥⑦⑧⑨⑩]", # 3순위: 동그라미 기호 (항 단위)
    r"\n\s*\d+\.",          # 4순위: '1.', '2.' (호 단위)
    r"\n\s*[가-하]\.",       # 5순위: '가.', '나.' (목 단위)
    "\n\n",                 # 6순위: 일반 단락 변경
    "\n",                   # 7순위: 줄바꿈
    r"\. ",                 # 8순위: 문장 끝 (정규식 이스케이프 추가)
    " ",                    # 9순위: 띄어쓰기
    ""                      # 10순위: 글자 단위 (필수)
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,           # 한 청크에 법령 한 조항이 충분히 들어가도록 크기를 넉넉히 잡습니다.
    chunk_overlap=150,        # 문맥이 끊기지 않도록 이전 청크와 150자 정도 겹치게 합니다.
    separators=custom_separators,
    is_separator_regex=True   # 정규표현식 사용 활성화
)

chunks = text_splitter.split_documents(documents)
print(f"총 {len(chunks)} 개의 텍스트 청크로 분할되었습니다.")

# --- 4. 벡터 변환 및 ChromaDB 저장 ---
print("3. 임베딩 모델 로드 및 벡터 DB 저장을 시작합니다...")

# 한국어 법령 데이터에 강한 임베딩 모델 사용
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# DB 생성 및 디스크에 저장
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"🎉 완료! 법령 맞춤형 벡터 DB가 '{persist_directory}' 폴더에 성공적으로 저장되었습니다.")