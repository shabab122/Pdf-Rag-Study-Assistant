"""Streamlit web interface for the PDF RAG Study Assistant."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from dotenv import load_dotenv

from src.config import COLLECTION_NAME, DATABASE_FOLDER, PDF_FOLDER, PROJECT_ROOT
from src.rag_answer import generate_answer, retrieve_context
from src.read_pdf import find_pdf_files
from src.store_embeddings import main as build_vector_database

MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024

st.set_page_config(
    page_title="PDF RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(PROJECT_ROOT / ".env")


def apply_styles() -> None:
    """Apply a clean, modern visual theme without external assets."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 8%, #e8edff 0, transparent 28%),
                radial-gradient(circle at 88% 2%, #e1f6ff 0, transparent 25%),
                #f7f9fc;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.88);
            border-right: 1px solid #e4e9f2;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .hero-card {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
            color: white;
            background: linear-gradient(135deg, #4338ca, #2563eb 58%, #0891b2);
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(37, 99, 235, 0.20);
        }

        .hero-card h1 {
            margin: 0 0 0.35rem 0;
            color: white;
            font-size: clamp(2rem, 4vw, 3rem);
        }

        .hero-card p {
            margin: 0;
            color: rgba(255, 255, 255, 0.88);
            font-size: 1.05rem;
        }

        [data-testid="stMetric"] {
            min-height: 112px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid #e5eaf2;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stChatMessage"] {
            margin: 0.75rem 0;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid #e5eaf2;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            font-weight: 600;
        }

        [data-testid="stChatInput"] {
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.10);
        }

        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api_key_is_ready() -> bool:
    """Return whether a non-placeholder Groq API key is configured."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(api_key and api_key != "replace_with_your_groq_api_key")


def get_index_chunk_count() -> int | None:
    """Return the number of indexed chunks, or None when no valid index exists."""
    if not DATABASE_FOLDER.exists():
        return None

    try:
        client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return None


def extract_sources(context: str) -> list[str]:
    """Extract unique filename/page labels from retrieved context."""
    sources = []

    for block in context.split("\n\n---\n\n"):
        first_line = block.splitlines()[0] if block.splitlines() else ""
        if not first_line.startswith("Source: ") or ", Page: " not in first_line:
            continue

        source, page = first_line.removeprefix("Source: ").rsplit(", Page: ", 1)
        label = f"{source} · page {page}"
        if label not in sources:
            sources.append(label)

    return sources


def choose_available_path(filename: str, content: bytes) -> Path:
    """Avoid silently overwriting a different local PDF with the same name."""
    safe_name = Path(filename).name.strip()
    if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
        raise ValueError("Only files with a .pdf extension are accepted.")

    target = PDF_FOLDER / safe_name
    if not target.exists() or target.read_bytes() == content:
        return target

    counter = 2
    while True:
        candidate = PDF_FOLDER / f"{target.stem}-{counter}{target.suffix}"
        if not candidate.exists() or candidate.read_bytes() == content:
            return candidate
        counter += 1


def save_uploaded_pdfs(uploaded_files: list[Any]) -> list[str]:
    """Validate and save uploaded PDFs inside the local data folder."""
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    saved_names = []

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()

        if len(content) > MAX_PDF_SIZE_BYTES:
            raise ValueError(f"{uploaded_file.name} is larger than the 50 MB limit.")
        if not content.startswith(b"%PDF-"):
            raise ValueError(f"{uploaded_file.name} does not appear to be a valid PDF.")

        target = choose_available_path(uploaded_file.name, content)
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)
        saved_names.append(target.name)

    return saved_names


def rebuild_index() -> tuple[bool, str]:
    """Run indexing and capture its terminal-style progress output."""
    output = io.StringIO()

    try:
        with redirect_stdout(output):
            exit_code = build_vector_database()
    except Exception as exc:
        return False, f"{output.getvalue()}\n{exc}".strip()

    return exit_code == 0, output.getvalue().strip()


def render_assistant_message(message: dict[str, Any]) -> None:
    """Render an answer, its source labels, and optional retrieved passages."""
    st.markdown(message["content"])

    sources = message.get("sources", [])
    if sources:
        st.caption("Sources: " + "  •  ".join(sources))

    context = message.get("context")
    if context:
        with st.expander("View retrieved PDF passages"):
            st.text(context)


apply_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Knowledge base")
    st.caption("Upload PDFs, build the index, then ask questions in the chat.")

    uploaded_files = st.file_uploader(
        "Upload PDF notes",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 50 MB per file. Scanned PDFs need OCR first.",
    )

    if st.button(
        "Save uploaded PDFs",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        try:
            saved = save_uploaded_pdfs(uploaded_files)
            st.success(f"Saved {len(saved)} PDF(s): {', '.join(saved)}")
            st.info("Build the knowledge base before asking questions.")
        except ValueError as exc:
            st.error(str(exc))

    if st.button("Build / rebuild knowledge base", type="primary", use_container_width=True):
        if not find_pdf_files():
            st.error("Upload at least one text-based PDF first.")
        else:
            with st.spinner("Reading PDFs and creating embeddings..."):
                success, build_output = rebuild_index()

            if success:
                st.success("Knowledge base is ready.")
            else:
                st.error("The knowledge base could not be built.")

            if build_output:
                with st.expander("Indexing details", expanded=not success):
                    st.code(build_output, language="text")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<p class='small-note'>Your PDFs and vector database remain inside "
        "this local project. Retrieved text is sent to Groq only when an answer "
        "is generated.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero-card">
        <h1>PDF RAG Study Assistant</h1>
        <p>Ask questions about your course PDFs and receive focused Banglish answers.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

pdf_files = find_pdf_files()
chunk_count = get_index_chunk_count()

metric_pdf, metric_chunks, metric_api = st.columns(3)
metric_pdf.metric("PDF files", len(pdf_files))
metric_chunks.metric("Indexed chunks", chunk_count if chunk_count is not None else "Not built")
metric_api.metric("Groq API", "Ready" if api_key_is_ready() else "Not configured")

if not pdf_files:
    st.info("Start by uploading a text-based PDF from the sidebar.")
elif not chunk_count:
    st.warning("PDFs are available, but the knowledge base has not been built yet.")

if not api_key_is_ready():
    st.warning("Add `GROQ_API_KEY` to the `.env` file before generating answers.")

st.subheader("Study chat")

if not st.session_state.messages:
    st.markdown(
        "Ask a focused question such as **What is a shell script?** or "
        "**Explain the main steps from chapter 2.**"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDFs...")

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not chunk_count:
            answer_message = {
                "role": "assistant",
                "content": "Please build the knowledge base from the sidebar first.",
            }
            st.warning(answer_message["content"])
        elif not api_key_is_ready():
            answer_message = {
                "role": "assistant",
                "content": "Please configure `GROQ_API_KEY` in `.env` first.",
            }
            st.warning(answer_message["content"])
        else:
            try:
                with st.spinner("Searching your PDFs and preparing the answer..."):
                    context = retrieve_context(question)
                    answer = generate_answer(question, context)

                answer_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": extract_sources(context),
                    "context": context,
                }
                render_assistant_message(answer_message)
            except Exception as exc:
                answer_message = {
                    "role": "assistant",
                    "content": f"I could not generate an answer: {exc}",
                }
                st.error(answer_message["content"])

    st.session_state.messages.append(answer_message)
