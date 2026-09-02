import argparse

import chromadb

from .config import COLLECTION_NAME, DATABASE_FOLDER
from .embedding_model import get_embedding_model


def search_pdf(question: str, result_count: int = 3) -> None:
    """Find the PDF chunks most related to a question."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if result_count <= 0:
        raise ValueError("result_count must be greater than zero.")
    if not DATABASE_FOLDER.exists():
        raise RuntimeError(
            "The vector database does not exist. Run: python -m src.store_embeddings"
        )

    model = get_embedding_model()

    # Convert the question into an embedding.
    question_embedding = model.encode(question).tolist()

    # Open the saved vector database.
    client = chromadb.PersistentClient(path=str(DATABASE_FOLDER))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "The PDF index is missing. Run: python -m src.store_embeddings"
        ) from exc

    stored_count = collection.count()
    if stored_count == 0:
        raise RuntimeError("The PDF index is empty. Rebuild it before searching.")

    # Find the most similar stored chunks.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(result_count, stored_count),
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\nQuestion: {question}\n")

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"--- Result {index} ---")
        print(f"Source: {metadata['source']} | Page: {metadata['page']}")
        print(f"Distance: {distance:.4f}")
        print(f"Text: {document}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search indexed PDF notes.")
    parser.add_argument("question", nargs="*", help="question to search for")
    parser.add_argument(
        "--results",
        type=int,
        default=3,
        help="maximum number of matching chunks (default: 3)",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask a question about the PDF: ").strip()

    try:
        search_pdf(question, args.results)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
