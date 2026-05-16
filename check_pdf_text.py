import os
from langchain_community.document_loaders import PyPDFLoader

pdf_directory = "./data/raw_data"
total_pages = 0
empty_pages = 0

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            total_pages += 1
            if not doc.page_content.strip():
                empty_pages += 1

print(f"총 페이지 수: {total_pages}")
print(f"텍스트가 비어있는 페이지 수: {empty_pages}")
if total_pages == empty_pages:
    print("경고: 모든 PDF 페이지에서 텍스트를 추출할 수 없습니다. 스캔된 이미지거나 암호가 걸려있을 수 있습니다.")
