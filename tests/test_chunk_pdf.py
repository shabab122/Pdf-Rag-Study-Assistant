import unittest

from src.chunk_pdf import create_chunks, split_text


class SplitTextTests(unittest.TestCase):
    def test_short_text_creates_one_chunk(self) -> None:
        self.assertEqual(split_text("short text", 50, 10), ["short text"])

    def test_chunks_overlap_without_duplicate_tail(self) -> None:
        self.assertEqual(
            split_text("abcdefghij", 5, 2),
            ["abcde", "defgh", "ghij"],
        )

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_text("text", 0, 0)

        with self.assertRaises(ValueError):
            split_text("text", 5, 5)


class CreateChunksTests(unittest.TestCase):
    def test_source_and_page_metadata_are_preserved(self) -> None:
        pages = [{"source": "notes.pdf", "page": 2, "text": "hello"}]

        chunks = create_chunks(pages)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["id"], "notes.pdf-page-2-chunk-1")
        self.assertEqual(chunks[0]["source"], "notes.pdf")
        self.assertEqual(chunks[0]["page"], 2)
        self.assertEqual(chunks[0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
