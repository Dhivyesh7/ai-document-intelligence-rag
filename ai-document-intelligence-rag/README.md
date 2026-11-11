# AI-Powered Document Intelligence (RAG)

A Streamlit application that lets users upload PDF documents and ask questions about their contents. It extracts page text, splits it into overlapping chunks, creates sentence embeddings, stores vectors in ChromaDB, retrieves relevant passages, and generates grounded answers through the Groq API.

> **Portfolio note:** This is an original starter implementation. Before presenting it as a completed project, run it, test it with your own PDFs, customize at least one feature, and understand the retrieval and generation flow.

## Features

- Upload one or more PDF files.
- Extract text while preserving filename and page metadata.
- Chunk text with overlap to retain context across boundaries.
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors in an in-memory ChromaDB collection.
- Retrieve the most relevant passages for each question.
- Generate answers with Groq and display source-page references.
- Keep a chat history for the current browser session.

## Architecture

`PDF upload -> pypdf extraction -> text chunking -> SentenceTransformer embeddings -> ChromaDB similarity search -> Groq LLM answer -> source references`

## Requirements

- Python 3.10 or newer
- A Groq API key: https://console.groq.com/keys
- Internet access the first time the embedding model is downloaded

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the API key

Copy `.env.example` to `.env` and add your key:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Never commit `.env` or publish your API key.

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Run tests

```bash
pytest -q
```

## Limitations and responsible use

- Scanned PDFs that contain only images may not yield extractable text; OCR is not included.
- Retrieval quality depends on document quality, chunk size, and embedding relevance.
- The LLM can still produce inaccurate answers. Check the displayed source passages before relying on an answer.
- Documents are held in an in-memory vector collection for the current app session; they are not a durable document store.
- Do not upload confidential or personal documents to an external LLM service without authorization.

## Suggested extensions

1. Add OCR for scanned documents.
2. Add a document comparison mode.
3. Add persistent ChromaDB storage and user-specific collections.
4. Add automated retrieval evaluation with a small set of question/answer test cases.

## Technology stack

Python, Streamlit, pypdf, Sentence Transformers, ChromaDB, Groq API, pytest.
