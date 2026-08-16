import unittest

from app.services.prompts import SYSTEM_PROMPT, build_rag_messages, format_contexts


class PromptTests(unittest.TestCase):
    def test_no_context_has_explicit_placeholder(self):
        self.assertEqual(
            format_contexts([]),
            "No relevant document context was retrieved.",
        )

    def test_context_format_preserves_source_metadata_and_content(self):
        formatted = format_contexts(
            [
                {
                    "filename": "policy.md",
                    "chunk_id": "doc-1",
                    "position": 2,
                    "chunk_index": 1,
                    "page_number": 7,
                    "score": 0.8,
                    "content": "Annual leave requires manager approval.",
                }
            ]
        )

        self.assertIn("Source: policy.md", formatted)
        self.assertIn("Chunk ID: doc-1", formatted)
        self.assertIn("Position: 2", formatted)
        self.assertIn("Chunk Index: 1", formatted)
        self.assertIn("Page Number: 7", formatted)
        self.assertIn("Annual leave requires manager approval.", formatted)

    def test_system_prompt_requires_grounded_no_answer_behavior(self):
        normalized = " ".join(SYSTEM_PROMPT.lower().split())

        self.assertIn("using only the provided document context", normalized)
        self.assertIn("does not contain a reliable answer", normalized)
        self.assertIn("do not invent facts", normalized)
        self.assertIn("directly support the answer", normalized)

    def test_rag_messages_separate_system_instruction_and_user_data(self):
        messages = build_rag_messages(
            "How is leave approved?",
            [
                {
                    "filename": "policy.md",
                    "chunk_id": "doc-1",
                    "position": 1,
                    "score": 0.9,
                    "content": "Managers approve annual leave.",
                }
            ],
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertIn("Question:\nHow is leave approved?", messages[1]["content"])
        self.assertIn("Retrieved context:", messages[1]["content"])
        self.assertIn("Managers approve annual leave.", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
