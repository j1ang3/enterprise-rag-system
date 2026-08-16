import unittest
from unittest.mock import patch

from app.services import vector_store
from scripts import qdrant_smoke_test


class QdrantSmokeScriptTests(unittest.TestCase):
    def test_skips_when_qdrant_is_not_reachable(self):
        with patch.object(
            qdrant_smoke_test,
            "parse_args",
            return_value=qdrant_smoke_test.argparse.Namespace(
                url="http://localhost:6333",
                api_key="",
                collection="enterprise_rag_smoke_test",
                require_running=False,
                keep_collection=False,
            ),
        ), patch.object(vector_store, "_qdrant_request", return_value=None), patch("builtins.print"):
            exit_code = qdrant_smoke_test.main()

        self.assertEqual(exit_code, 0)

    def test_require_running_fails_when_qdrant_is_not_reachable(self):
        with patch.object(
            qdrant_smoke_test,
            "parse_args",
            return_value=qdrant_smoke_test.argparse.Namespace(
                url="http://localhost:6333",
                api_key="",
                collection="enterprise_rag_smoke_test",
                require_running=True,
                keep_collection=False,
            ),
        ), patch.object(vector_store, "_qdrant_request", return_value=None), patch("builtins.print"):
            exit_code = qdrant_smoke_test.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
