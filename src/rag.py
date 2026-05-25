# src/rag.py
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama # <== HuggingFace 대신 OllamaEmbeddings 사용
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, SYSTEM_PROMPT, OLLAMA_MODEL, TEMPERATURE, format_docs

def get_chroma_retriever(search_k=4):
    # Ollama 전용 임베딩으로 변경
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": search_k})

def create_rag_chain(model_name=OLLAMA_MODEL, temperature=TEMPERATURE, search_k=4):
    retriever = get_chroma_retriever(search_k)
    
    llm = ChatOllama(
        model=model_name,
        temperature=temperature, 
        base_url="http://localhost:11434"
    )
    
    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever