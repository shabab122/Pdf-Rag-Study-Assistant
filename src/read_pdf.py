from pathlib import Path

from pypdf import PdfReader


def read_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract non-empty text from a PDF while preserving page metadata."""
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def main() -> None:
    pdf_folder = Path("data")
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF found. Put at least one PDF inside the data folder.")
        return

    for pdf_path in pdf_files:
        pages = read_pdf_pages(pdf_path)
        print(f"PDF: {pdf_path.name}")
        print(f"Pages with text: {len(pages)}")
        for page in pages[:2]:
            print(f"\n--- Page {page['page']} ---\n{page['text'][:1000]}")


if __name__ == "__main__":
    main()
