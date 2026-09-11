try:
    from .retrieval import DEFAULT_RESULT_COUNT, retrieve
except ImportError:  # Supports: python src/search_pdf.py
    from retrieval import DEFAULT_RESULT_COUNT, retrieve


def search_pdf(question: str, result_count: int = DEFAULT_RESULT_COUNT) -> None:
    response = retrieve(question, result_count=result_count)
    print(f"Distance threshold: {response.threshold:.2f} (lower distance is better)")

    if not response.candidates:
        print("No chunks are stored in ChromaDB. Run store_embeddings.py first.")
        return

    for index, chunk in enumerate(response.candidates, start=1):
        status = "ACCEPTED" if chunk in response.chunks else "REJECTED"
        print(f"\n--- Result {index}: {status} ---")
        print(f"Source: {chunk.source} | Page: {chunk.page}")
        print(f"Distance: {chunk.distance:.4f}")
        print(f"Text: {chunk.text[:1000]}")

    if not response.has_context:
        print("\nNo result passed the similarity threshold.")


def main() -> None:
    question = input("Question: ").strip()
    if question:
        search_pdf(question)


if __name__ == "__main__":
    main()
