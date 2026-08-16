import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr, ValidationError

from app.auth.tokens import AuthenticationConfigurationError, require_jwt_secret
from app.core.config import Settings, settings
from app.db.session import (
    DatabaseConfigurationError,
    require_database_url,
    require_test_database_url,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
RENDER_BLUEPRINT_PATH = PROJECT_ROOT / "render.yaml"
DEPLOY_DATABASE_WORKFLOW_PATH = (
    PROJECT_ROOT / ".github" / "workflows" / "deploy-database.yml"
)
ASSIGNMENT_PATTERN = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=(.*)$")
COMPOSE_INPUT_PATTERN = re.compile(r"\$\{(COMPOSE_[A-Z0-9_]+)")


def _documented_assignments() -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        match = ASSIGNMENT_PATTERN.match(line)
        if match:
            assignments[match.group(1)] = match.group(2)
    return assignments


def _canonical_environment_name(field_name: str, field: object) -> str:
    validation_alias = getattr(field, "validation_alias", None)
    choices = getattr(validation_alias, "choices", None)
    if choices:
        return str(choices[0])
    if isinstance(validation_alias, str):
        return validation_alias
    return field_name.upper()


class SettingsContractTests(unittest.TestCase):
    def test_defaults_are_stable_without_environment_or_dotenv(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configured = Settings(_env_file=None)

        self.assertEqual(configured.app_name, "Enterprise RAG Knowledge Base")
        self.assertEqual(configured.app_version, "0.1.0")
        self.assertEqual(configured.jwt_access_token_expire_minutes, 30)
        self.assertEqual(configured.rag_security_mode, "layered")
        self.assertEqual(configured.database_url, "")
        self.assertEqual(configured.test_database_url, "")
        self.assertEqual(configured.jwt_secret_key.get_secret_value(), "")

    def test_canonical_and_legacy_llm_environment_names_are_supported(self) -> None:
        with patch.dict(
            os.environ,
            {"APP_NAME": "Configured Application", "LLM_MODEL": "gemma3:4b"},
            clear=True,
        ):
            configured = Settings(_env_file=None)
        self.assertEqual(configured.app_name, "Configured Application")
        self.assertEqual(configured.llm_model, "gemma3:4b")

        with patch.dict(os.environ, {"OLLAMA_MODEL": "qwen3:8b"}, clear=True):
            configured_from_alias = Settings(_env_file=None)
        self.assertEqual(configured_from_alias.llm_model, "qwen3:8b")

    def test_invalid_bounded_or_enumerated_values_fail_validation(self) -> None:
        invalid_environments = (
            {"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "0"},
            {"JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "1441"},
            {"RAG_SECURITY_MODE": "unsafe"},
            {"LLM_TEMPERATURE": "2.1"},
            {"LLM_MAX_RETRIES": "6"},
        )

        for invalid_environment in invalid_environments:
            with self.subTest(environment=tuple(invalid_environment)):
                with patch.dict(os.environ, invalid_environment, clear=True):
                    with self.assertRaises(ValidationError):
                        Settings(_env_file=None)

    def test_secret_values_are_masked_and_auth_fails_closed(self) -> None:
        synthetic_secret = "configuration-test-secret-value-12345"
        self.assertNotIn(synthetic_secret, repr(SecretStr(synthetic_secret)))

        with patch.object(settings, "jwt_secret_key", SecretStr("")):
            with self.assertRaises(AuthenticationConfigurationError):
                require_jwt_secret()
        with patch.object(settings, "jwt_secret_key", SecretStr("too-short")):
            with self.assertRaises(AuthenticationConfigurationError):
                require_jwt_secret()
        with patch.object(settings, "jwt_secret_key", SecretStr(synthetic_secret)):
            self.assertEqual(require_jwt_secret(), synthetic_secret)

    def test_database_configuration_has_no_implicit_fallback(self) -> None:
        with patch.object(settings, "database_url", ""):
            with self.assertRaises(DatabaseConfigurationError):
                require_database_url()

        with patch.object(settings, "database_url", "postgresql://user:pw@db/app"):
            with patch.object(settings, "test_database_url", ""):
                with self.assertRaises(DatabaseConfigurationError):
                    require_test_database_url()
            with patch.object(
                settings,
                "test_database_url",
                "postgresql://user:other@db/app",
            ):
                with self.assertRaises(DatabaseConfigurationError):
                    require_test_database_url()
            with patch.object(
                settings,
                "test_database_url",
                "postgresql://user:pw@db/app_test",
            ):
                self.assertTrue(require_test_database_url().endswith("/app_test"))


class EnvironmentExampleContractTests(unittest.TestCase):
    def test_example_documents_every_canonical_application_setting(self) -> None:
        documented_names = set(_documented_assignments())
        canonical_names = {
            _canonical_environment_name(field_name, field)
            for field_name, field in Settings.model_fields.items()
        }
        self.assertSetEqual(canonical_names - documented_names, set())

    def test_example_documents_every_compose_input(self) -> None:
        compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
        compose_inputs = set(COMPOSE_INPUT_PATTERN.findall(compose_text))
        documented_names = set(_documented_assignments())
        self.assertSetEqual(compose_inputs - documented_names, set())

    def test_example_has_no_duplicate_or_stale_assignments(self) -> None:
        example_lines = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines()
        documented_names = [
            match.group(1)
            for line in example_lines
            if (match := ASSIGNMENT_PATTERN.match(line))
        ]
        canonical_names = {
            _canonical_environment_name(field_name, field)
            for field_name, field in Settings.model_fields.items()
        }
        compose_inputs = set(
            COMPOSE_INPUT_PATTERN.findall(COMPOSE_PATH.read_text(encoding="utf-8"))
        )

        self.assertEqual(len(documented_names), len(set(documented_names)))
        self.assertSetEqual(set(documented_names) - canonical_names - compose_inputs, set())

    def test_compose_maps_inputs_to_canonical_backend_settings(self) -> None:
        compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
        expected_mappings = (
            "DATABASE_URL: ${COMPOSE_DATABASE_URL:?Set COMPOSE_DATABASE_URL}",
            "JWT_SECRET_KEY: ${COMPOSE_JWT_SECRET_KEY:?Set COMPOSE_JWT_SECRET_KEY}",
            "LLM_PROVIDER: ${COMPOSE_LLM_PROVIDER:-ollama}",
            "LLM_API_KEY: ${COMPOSE_LLM_API_KEY:-ollama}",
            "LLM_BASE_URL: ${COMPOSE_LLM_BASE_URL:-http://host.docker.internal:11434/v1}",
            "LLM_MODEL: ${COMPOSE_LLM_MODEL:-gemma3:4b}",
        )
        for mapping in expected_mappings:
            with self.subTest(mapping=mapping):
                self.assertIn(mapping, compose_text)

    def test_sensitive_example_values_are_blank(self) -> None:
        assignments = _documented_assignments()
        for name in (
            "DATABASE_URL",
            "TEST_DATABASE_URL",
            "JWT_SECRET_KEY",
            "COMPOSE_POSTGRES_PASSWORD",
            "COMPOSE_DATABASE_URL",
            "COMPOSE_JWT_SECRET_KEY",
        ):
            with self.subTest(name=name):
                self.assertEqual(assignments[name], "")

        example_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8").lower()
        for unsafe_placeholder in ("change-me", "password123", "supersecret"):
            self.assertNotIn(unsafe_placeholder, example_text)


class DeploymentContractTests(unittest.TestCase):
    def test_render_blueprint_is_explicitly_free_and_ci_gated(self) -> None:
        blueprint = RENDER_BLUEPRINT_PATH.read_text(encoding="utf-8")

        for contract in (
            "runtime: docker",
            "plan: free",
            "region: singapore",
            "autoDeployTrigger: checksPass",
            "healthCheckPath: /health",
            'value: "8000"',
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, blueprint)

        self.assertNotIn("preDeployCommand:", blueprint)
        self.assertNotIn("databases:", blueprint)

    def test_render_blueprint_keeps_real_secrets_out_of_git(self) -> None:
        blueprint = RENDER_BLUEPRINT_PATH.read_text(encoding="utf-8")

        self.assertRegex(blueprint, r"key: DATABASE_URL\s+sync: false")
        self.assertRegex(blueprint, r"key: JWT_SECRET_KEY\s+generateValue: true")
        self.assertNotIn("postgresql+psycopg://", blueprint)
        self.assertNotIn("qwen3:8b", blueprint)

    def test_deployment_migration_is_manual_direct_and_explicit(self) -> None:
        workflow = DEPLOY_DATABASE_WORKFLOW_PATH.read_text(encoding="utf-8")

        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+push:")
        self.assertIn("secrets.NEON_DIRECT_DATABASE_URL", workflow)
        self.assertIn("Migrations require the direct Neon endpoint", workflow)
        self.assertIn("python -m alembic upgrade head", workflow)
        self.assertIn("python -m alembic current --check-heads", workflow)
        self.assertIn("Expected PostgreSQL 18", workflow)


if __name__ == "__main__":
    unittest.main()
