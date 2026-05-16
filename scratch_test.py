import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_directory = "./data/raw_data"
documents = []
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        break  # Just load one for testing

print("Loaded pages:", len(documents))
if len(documents) > 0:
    print("First page length:", len(documents[0].page_content))
    print("First page content snippet:", repr(documents[0].page_content[:100]))

custom_separators = [
    r"\n제\s*\d+\s*장",
    r"\n제\s*\d+\s*조",
    r"\n\s*[①②③④⑤⑥⑦⑧⑨⑩]",
    r"\n\s*\d+\.",
    r"\n\s*[가-하]\.",
    "\n\n",
    "\n",
    ". ",
    " "
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=custom_separators,
    is_separator_regex=True
)

chunks = text_splitter.split_documents(documents)
print("Chunks:", len(chunks))
