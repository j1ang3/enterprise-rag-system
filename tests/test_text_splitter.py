import unittest

from app.services.text_splitter import split_text


class TextSplitterTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_text(""), [])

    def test_fixed_size_chunks_preserve_overlap(self):
        chunks = split_text("abcdefghij", chunk_size=6, chunk_overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij"])
        self.assertEqual(chunks[0][-2:], chunks[1][:2])

    def test_chunks_respect_maximum_size(self):
        chunks = split_text(
            "First sentence. Second sentence. Third sentence.",
            chunk_size=24,
            chunk_overlap=4,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 24 for chunk in chunks))

    def test_chunk_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            split_text("text", chunk_size=0, chunk_overlap=0)

    def test_chunk_overlap_must_not_be_negative(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            split_text("text", chunk_size=10, chunk_overlap=-1)

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            split_text("text", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
