from .config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_FOLDER
from .read_pdf import find_pdf_files, read_pdf_pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(pages: list[dict]) -> list[dict]:
    """Create chunks and preserve their source and page metadata."""
    chunks = []

    for page in pages:
        text_chunks = split_text(
            text=page["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_number, text in enumerate(text_chunks, start=1):
            chunks.append(
                {
                    "id": f"{page['source']}-page-{page['page']}-chunk-{chunk_number}",
                    "source": page["source"],
                    "page": page["page"],
                    "text": text,
                }
            )

    return chunks


def main() -> int:
    pdf_files = find_pdf_files()

    if not pdf_files:
        print(f"No PDF found. Put at least one PDF inside: {PDF_FOLDER}")
        return 1

    chunks = []
    for pdf_path in pdf_files:
        pages = read_pdf_pages(pdf_path)
        chunks.extend(create_chunks(pages))

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("No extractable text was found. Scanned PDFs require OCR first.")
        return 1

    for chunk in chunks[:3]:
        print(f"\n--- {chunk['id']} ---")
        print(chunk["text"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
