from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


CLIENT = TestClient(app)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_OPERATIONS = {
    ("get", "/"),
    ("get", "/health"),
    ("post", "/auth/register"),
    ("post", "/auth/login"),
}

PROTECTED_OPERATIONS = {
    ("get", "/auth/me"),
    ("post", "/documents/upload"),
    ("get", "/documents/"),
    ("post", "/documents/{document_id}/shares"),
    ("get", "/documents/{document_id}/shares"),
    ("delete", "/documents/{document_id}/shares/{user_id}"),
    ("get", "/documents/{document_id}/preview"),
    ("get", "/documents/{document_id}/chunks"),
    ("post", "/search"),
    ("post", "/search/hybrid"),
    ("post", "/search/hybrid/fused"),
    ("post", "/chat/"),
}


def _openapi() -> dict:
    response = CLIENT.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_exposes_the_current_sixteen_operation_contract() -> None:
    schema = _openapi()
    actual_operations = {
        (method, path)
        for path, path_item in schema["paths"].items()
        for method in path_item
        if method in {"get", "post", "put", "patch", "delete"}
    }

    assert len(schema["paths"]) == 15
    assert actual_operations == PUBLIC_OPERATIONS | PROTECTED_OPERATIONS


def test_openapi_security_matches_public_and_bearer_protected_routes() -> None:
    schema = _openapi()
    security_schemes = schema["components"]["securitySchemes"]

    assert security_schemes["HTTPBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }
    for method, path in PROTECTED_OPERATIONS:
        assert {"HTTPBearer": []} in schema["paths"][path][method]["security"]
    for method, path in PUBLIC_OPERATIONS:
        assert "security" not in schema["paths"][path][method]


def test_openapi_documents_success_models_errors_and_request_examples() -> None:
    schema = _openapi()

    root_response = schema["paths"]["/"]["get"]["responses"]["200"]
    health_response = schema["paths"]["/health"]["get"]["responses"]["200"]
    assert root_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RootResponse"
    )
    assert health_response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/HealthResponse"
    )

    assert {"401", "403", "404", "503"}.issubset(
        schema["paths"]["/documents/{document_id}/preview"]["get"]["responses"]
    )
    assert {"400", "401", "409", "422", "500", "503"}.issubset(
        schema["paths"]["/documents/upload"]["post"]["responses"]
    )

    for component_name in (
        "RegisterRequest",
        "LoginRequest",
        "GrantDocumentAccessRequest",
        "SemanticSearchRequest",
        "HybridSearchRequest",
        "RankFusionSearchRequest",
        "ChatRequest",
    ):
        assert schema["components"]["schemas"][component_name]["examples"]


def test_human_api_guide_mentions_every_current_operation() -> None:
    api_guide = (REPOSITORY_ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    for method, path in PUBLIC_OPERATIONS | PROTECTED_OPERATIONS:
        assert f"`{method.upper()} {path}`" in api_guide

    assert "owner" in api_guide.lower()
    assert "acl" in api_guide.lower()
    assert "local_fallback" in api_guide
