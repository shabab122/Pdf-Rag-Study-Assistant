from typing import Any

import chromadb

from .chunk_pdf import create_chunks
from .config import COLLECTION_NAME, DATABASE_FOLDER, PDF_FOLDER
from .embedding_model import get_embedding_model
from .read_pdf import find_pdf_files, read_pdf_pages

BATCH_SIZE = 1_000


def _collection_names(client: Any) -> set[str]:
    """Return collection names across supported ChromaDB result formats."""
    return {
        collection.name if hasattr(collection, "name") else str(collection)
        for collection in client.list_collections()
    }


def main() -> int:
    pdf_files = find_pdf_files()

    if not pdf_files:
        print(f"No PDF found. Put at least one PDF inside: {PDF_FOLDER}")
        return 1

    # 1. Read every PDF and split all pages into chunks
    chunks = []

    for pdf_path in pdf_files:
        pages = read_pdf_pages(pdf_path)
        pdf_chunks = create_chunks(pages)
        chunks.extend(pdf_chunks)

        print(f"Read {pdf_path.name}: {len(pdf_chunks)} chunks")

    if not chunks:
        print("No extractable text was found. Scanned PDFs require OCR first.")
        return 1

    # 2. Load the embedding model
    print("Loading embedding model...")
    model = get_embedding_model()

    # 3. Convert chunk text into embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).tolist()

    # 4. Create/open a local ChromaDB database
    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))

    # Remove the old collection so rerunning does not create duplicate chunks.
    if COLLECTION_NAME in _collection_names(client):
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(COLLECTION_NAME)

    # 5. Store embeddings, text, and source metadata
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        end = start + len(batch)

        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                }
                for chunk in batch
            ],
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    print(f"Database folder created: {DATABASE_FOLDER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
