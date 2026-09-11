from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from .chunk_pdf import create_chunks
    from .read_pdf import read_pdf_pages
except ImportError:  # Supports: python src/store_embeddings.py
    from chunk_pdf import create_chunks
    from read_pdf import read_pdf_pages


PDF_FOLDER = Path("data")
DATABASE_FOLDER = Path("chroma_db")
COLLECTION_NAME = "course_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_index() -> tuple[int, dict[str, int]]:
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF found. Put at least one PDF inside the data folder.")

    chunks: list[dict] = []
    chunk_counts: dict[str, int] = {}
    for pdf_path in pdf_files:
        pdf_chunks = create_chunks(read_pdf_pages(pdf_path))
        chunks.extend(pdf_chunks)
        chunk_counts[pdf_path.name] = len(pdf_chunks)
        print(f"Read {pdf_path.name}: {len(pdf_chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).tolist()

    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as error:
        if "does not exist" not in str(error).lower():
            raise

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        # Include the source and global position so chunks from different PDFs
        # cannot accidentally reuse the same Chroma ID.
        ids=[f"{chunk['source']}::{index}" for index, chunk in enumerate(chunks)],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in chunks
        ],
    )

    return len(chunks), chunk_counts


def main() -> None:
    total_chunks, _ = build_index()
    print(f"Stored {total_chunks} chunks in ChromaDB.")
    print(f"Database folder created: {DATABASE_FOLDER}")


if __name__ == "__main__":
    main()
