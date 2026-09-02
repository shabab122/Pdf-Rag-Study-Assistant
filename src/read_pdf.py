from pathlib import Path

from pypdf import PdfReader

from .config import PDF_FOLDER


def find_pdf_files() -> list[Path]:
    """Return local PDFs in a deterministic order."""
    return sorted(PDF_FOLDER.glob("*.pdf"), key=lambda path: path.name.casefold())


def read_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text and page numbers from one PDF."""
    try:
        reader = PdfReader(pdf_path)
        pages = []

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
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {pdf_path.name}") from exc

    return pages


def main() -> int:
    pdf_files = find_pdf_files()

    if not pdf_files:
        print(f"No PDF found. Put at least one PDF inside: {PDF_FOLDER}")
        return 1

    pdf_path = pdf_files[0]
    pages = read_pdf_pages(pdf_path)

    print(f"PDF: {pdf_path.name}")
    print(f"Pages with text: {len(pages)}")

    for page in pages[:2]:
        preview = page["text"][:300].replace("\n", " ")
        print(f"\n--- Page {page['page']} ---")
        print(preview)

    if not pages:
        print("\nNo extractable text was found. Scanned PDFs require OCR first.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
