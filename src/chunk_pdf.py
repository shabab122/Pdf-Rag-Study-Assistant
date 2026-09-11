from pathlib import Path

try:
    from .read_pdf import read_pdf_pages
except ImportError:  # Supports: python src/chunk_pdf.py
    from read_pdf import read_pdf_pages


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def create_chunks(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    chunk_number = 0

    for page in pages:
        for text_chunk in split_text(page["text"]):
            chunks.append(
                {
                    "id": f"chunk-{chunk_number}",
                    "source": page["source"],
                    "page": page["page"],
                    "text": text_chunk,
                }
            )
            chunk_number += 1

    return chunks


def main() -> None:
    pdf_folder = Path("data")
    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print("No PDF found. Put a PDF inside the data folder.")
        return

    for pdf_path in pdf_files:
        chunks = create_chunks(read_pdf_pages(pdf_path))
        print(f"{pdf_path.name}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()
