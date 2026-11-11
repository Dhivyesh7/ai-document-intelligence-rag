from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rag_utils import chunk_text

load_dotenv()

APP_TITLE = "AI-Powered Document Intelligence"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4


@st.cache_resource(show_spinner="Loading the sentence-embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_chunks(files: list[Any]) -> list[dict[str, Any]]:
    """Extract PDF page text and split it into metadata-tagged chunks."""
    extracted: list[dict[str, Any]] = []
    for uploaded_file in files:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            for chunk_number, text in enumerate(chunk_text(page_text), start=1):
                extracted.append(
                    {
                        "text": text,
                        "filename": uploaded_file.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    }
                )
    return extracted


def create_vector_collection(chunks: list[dict[str, Any]]) -> Any:
    """Build an in-memory ChromaDB collection for the uploaded PDFs."""
    model = load_embedding_model()
    texts = [item["text"] for item in chunks]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="uploaded_documents",
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    metadatas = [
        {"filename": item["filename"], "page": item["page"], "chunk": item["chunk"]}
        for item in chunks
    ]
    collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    return collection


def retrieve_context(question: str, collection: Any) -> list[dict[str, Any]]:
    model = load_embedding_model()
    query_vector = model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    retrieved: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append({"text": document, "metadata": metadata, "distance": distance})
    return retrieved


def generate_answer(question: str, passages: list[dict[str, Any]]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")

    context_parts = []
    for index, item in enumerate(passages, start=1):
        meta = item["metadata"]
        context_parts.append(
            f"[Source {index}: {meta['filename']}, page {meta['page']}]\n{item['text']}"
        )
    context = "\n\n".join(context_parts)

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document-question-answering assistant. "
                    "Answer using only the supplied context. If the context does not contain "
                    "the answer, say that the uploaded documents do not provide enough information. "
                    "Do not invent facts, numbers, quotations, or page references. "
                    "Mention the source filename and page when useful."
                ),
            },
            {
                "role": "user",
                "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
    )
    return completion.choices[0].message.content or "I could not generate an answer."


st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
st.title("📄 AI-Powered Document Intelligence")
st.caption("Ask questions about PDF documents using retrieval-augmented generation (RAG).")

with st.sidebar:
    st.header("Document workspace")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs are supported. Scanned image-only PDFs need OCR.",
    )
    index_button = st.button("Process documents", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.caption(f"Groq model: `{DEFAULT_MODEL}`")

if "collection" not in st.session_state:
    st.session_state.collection = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_count" not in st.session_state:
    st.session_state.document_count = 0

if index_button:
    if not uploaded_files:
        st.warning("Upload at least one PDF before processing.")
    else:
        try:
            with st.spinner("Extracting text and creating the search index..."):
                chunks = extract_pdf_chunks(uploaded_files)
                if not chunks:
                    st.error("No selectable text was found. The PDFs may be scanned images.")
                else:
                    st.session_state.collection = create_vector_collection(chunks)
                    st.session_state.document_count = len(chunks)
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(chunks)} text chunks from {len(uploaded_files)} PDF file(s).")
        except Exception as exc:  # Streamlit should show actionable errors, not crash silently.
            st.error(f"Could not process the PDFs: {exc}")

if st.session_state.collection is None:
    st.info("Upload PDF files in the sidebar and select **Process documents** to begin.")
else:
    st.success(f"Knowledge base ready: {st.session_state.document_count} chunks indexed.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved source passages"):
                    for source in message["sources"]:
                        meta = source["metadata"]
                        st.markdown(f"**{meta['filename']} — page {meta['page']}**")
                        st.caption(source["text"][:700])

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    sources = retrieve_context(question, st.session_state.collection)
                    answer = generate_answer(question, sources)
                st.markdown(answer)
                with st.expander("Retrieved source passages"):
                    for source in sources:
                        meta = source["metadata"]
                        st.markdown(f"**{meta['filename']} — page {meta['page']}**")
                        st.caption(source["text"][:700])
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"The question could not be answered: {exc}")
