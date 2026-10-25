# Agentic RAG Model

A document question-answering application built with Python, LangGraph, LangChain, OpenAI, FAISS, and Streamlit. The system processes uploaded PDF documents, retrieves relevant passages, rewrites follow-up questions using conversation history, and generates answers grounded in retrieved evidence.

## Features

- **PDF ingestion:** Upload and process PDF documents.
- **Document chunking:** Split extracted text into overlapping chunks for retrieval.
- **Semantic retrieval:** Generate embeddings using OpenAI and store document vectors in FAISS.
- **Query rewriting:** Convert follow-up questions into standalone retrieval queries using conversation history.
- **Evidence-grounded generation:** Generate answers using retrieved document content.
- **Page references:** Include source page numbers in generated answers when page metadata is available.
- **Interactive interface:** Upload documents and ask questions through a Streamlit chat interface.
- **Session management:** Maintain chat history and support starting a new chat.
- **Structured outputs:** Use Pydantic schemas to structure query-rewriting responses.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Streamlit | User interface |
| LangGraph | Orchestrating the RAG workflow |
| LangChain | LLM integration and document processing |
| OpenAI GPT-4o mini | Query rewriting and answer generation |
| OpenAI text-embedding-3-small | Text embeddings |
| FAISS | Vector storage and similarity search |
| PyPDFLoader | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| Pydantic | Structured model outputs |
| aiofiles | Asynchronous file saving |

## Architecture

The application follows a sequential retrieval-augmented generation workflow.

```text
PDF Upload
    |
    v
Save PDF
    |
    v
Extract Document Text
    |
    v
Split into Chunks
    |
    v
Generate Embeddings
    |
    v
Build FAISS Retriever
    |
    v
User Question
    |
    v
Query Rewriting
    |
    v
Retrieve Relevant Documents
    |
    v
Generate Evidence-Grounded Answer
    |
    v
Display Answer in Streamlit
```

The LangGraph workflow consists of three nodes:

1. **Query Rewriter:** Rewrites the current question into a standalone query, using the previous question when necessary.
2. **Document Retriever:** Retrieves the most relevant document chunks using the rewritten query.
3. **Answer Generator:** Uses the retrieved context to generate a concise answer and reference available source pages.

## Project Structure

```text
rag_model/
├── app.py
├── graph.py
├── setup.py
├── .env
├── data/
└── src/
    └── rag_model/
        ├── __init__.py
        ├── components/
        │   ├── __init__.py
        │   ├── agent_state.py
        │   ├── generate_answer.py
        │   ├── get_docs.py
        │   ├── models.py
        │   ├── query_rewriter.py
        │   └── retriever.py
        └── utils/
            ├── __init__.py
            └── common.py
```

The `data/` directory is created when a PDF is saved. It is used for uploaded documents at runtime.

## Prerequisites

- Python 3.11 or later
- An OpenAI API key with access to the required models
- pip
- Git (optional)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RawatRahul14/rag_model.git
cd rag_model
```


### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv/Scripts/activate
```

### 3. Install the project

```powershell
pip install -r requirements.txt
```


### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY = your_openai_api_key
```


### 5. Run the application

From the project root, run:

```powershell
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

## Usage

1. Launch the application.
2. Upload a PDF through the sidebar.
3. Submit the file and wait for document processing and indexing to complete.
4. Ask a question about the uploaded document.
5. Ask follow-up questions to test contextual query rewriting.
6. Review the generated answer and its available page references.
7. Start a new chat when you want to clear the conversation history.


## Configuration

The current implementation uses the following retrieval and model settings:

| Setting | Value |
|---|---|
| Chat model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Retrieved chunks | 4 |
| LLM temperature | 0 |

These settings are defined in the corresponding Python modules and can be adjusted as needed.