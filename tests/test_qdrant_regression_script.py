import argparse
import unittest
from unittest.mock import patch

from scripts import run_qdrant_regression


class QdrantRegressionScriptTests(unittest.TestCase):
    def test_skips_when_qdrant_is_not_reachable(self):
        args = argparse.Namespace(
            url="http://localhost:6333",
            api_key="",
            collection_prefix="enterprise_rag_regression",
            require_running=False,
            keep_collections=False,
        )

        with patch.object(run_qdrant_regression, "parse_args", return_value=args), patch.object(
            run_qdrant_regression,
            "qdrant_is_reachable",
            return_value=False,
        ), patch("builtins.print"):
            exit_code = run_qdrant_regression.main()

        self.assertEqual(exit_code, 0)

    def test_require_running_fails_when_qdrant_is_not_reachable(self):
        args = argparse.Namespace(
            url="http://localhost:6333",
            api_key="",
            collection_prefix="enterprise_rag_regression",
            require_running=True,
            keep_collections=False,
        )

        with patch.object(run_qdrant_regression, "parse_args", return_value=args), patch.object(
            run_qdrant_regression,
            "qdrant_is_reachable",
            return_value=False,
        ), patch("builtins.print"):
            exit_code = run_qdrant_regression.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
