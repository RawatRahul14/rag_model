# === Modules ===
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Retriever ===
def create_retriever(file_path: Path):
    ## === Load PDF ===
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    ## === Split Documents ===
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("The PDF contains no extractable text.")

    ## === Create Embeddings ===
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )

    ## === Create Vector Store ===
    vector_store = FAISS.from_documents(
        documents = chunks,
        embedding = embeddings,
    )

    ## === Return Retriever ===
    return vector_store.as_retriever(
        search_kwargs = {"k": 4}
    )