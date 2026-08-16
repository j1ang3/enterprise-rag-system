# Enterprise RAG System API Guide

## 1. Overview

The live OpenAPI document is the machine-readable source of truth for request and
response schemas. This guide explains the intended workflow, authentication and
authorization rules, retrieval modes, and operational limitations.

- Local base URL: `http://127.0.0.1:8000`
- Public demo base URL: `https://enterprise-rag-api-j1ang3.onrender.com`
- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`

Render Free instances can sleep when idle, so the first public request may take
longer than later requests.

## 2. Response and error shapes

Successful business responses use this envelope:

```json
{"success": true, "data": {}, "message": "operation completed"}
```

Handled HTTP errors use FastAPI's detail shape:

```json
{"detail": "Human-readable explanation."}
```

Request validation failures use FastAPI's structured `422` response. Important
status codes are:

- `400` invalid upload or other malformed business input;
- `401` missing, invalid, expired, or stale Bearer credentials;
- `403` authenticated user lacks document or ACL-management permission;
- `404` document, user, share, extracted text, or chunks do not exist;
- `409` duplicate user, conflicting metadata, or duplicate/invalid share;
- `422` request schema or field validation failed;
- `500` local extraction or persistence failed;
- `503` PostgreSQL authorization, metadata storage, embeddings, or retrieval is unavailable.

## 3. Authentication and authorization

Register with `POST /auth/register`, then obtain a JWT from `POST /auth/login`.
Send the returned token on every protected request:

```text
Authorization: Bearer <access-token>
```

`GET /auth/me` resolves the current PostgreSQL user. The login response contains
`access_token`, `token_type`, and the configured lifetime as `expires_in` seconds.
The default lifetime is 30 minutes, but deployment configuration may change it.

The JWT carries identity, not document permissions. At every document, search, or
chat request, the server evaluates current PostgreSQL ownership and read ACLs:

```text
Bearer JWT -> current user -> ownership + active ACL -> allowed document IDs
```

Owners have implicit read access. Explicitly shared readers have read access.
Everyone else is denied by default. Revoking a grant takes effect on the reader's
next request without replacing their JWT.

## 4. System and authentication endpoints

| Method and path | Purpose | Authentication |
| --- | --- | --- |
| `GET /` | Application discovery message | Public |
| `GET /health` | Process-level health check | Public |
| `POST /auth/register` | Register a PostgreSQL-backed user | Public |
| `POST /auth/login` | Verify credentials and issue a Bearer JWT | Public |
| `GET /auth/me` | Resolve the authenticated user | Bearer |

Registration and login examples use synthetic credentials only:

```bash
curl.exe -X POST "http://127.0.0.1:8000/auth/register" -H "Content-Type: application/json" -d "{\"username\":\"demo-user\",\"password\":\"synthetic-password-123\"}"
```

```bash
curl.exe -X POST "http://127.0.0.1:8000/auth/login" -H "Content-Type: application/json" -d "{\"username\":\"demo-user\",\"password\":\"synthetic-password-123\"}"
```

## 5. Document ingestion and reads

| Method and path | Purpose | Authentication |
| --- | --- | --- |
| `POST /documents/upload` | Extract, chunk, index, and register an owned document | Bearer |
| `GET /documents/` | List only owned and explicitly shared documents | Bearer |
| `GET /documents/{document_id}/preview` | Read an authorized text preview | Bearer + read access |
| `GET /documents/{document_id}/chunks` | Read authorized raw chunk content | Bearer + read access |

Upload is `multipart/form-data` with a `file` field. Supported extensions are
`.txt`, `.md`, `.pdf`, and `.docx`; the maximum file size is 10 MiB. A successful
upload returns the document ID, original filename, content type, storage paths,
chunk count, and preview. Storage paths are diagnostic development information and
should not be treated as stable public URLs.

```bash
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" -H "Authorization: Bearer <access-token>" -F "file=@C:\\path\\to\\synthetic-policy.txt"
```

The list is filtered server-side; the API never returns a global list for clients
to filter. Knowing a document ID does not bypass preview or chunk ACL checks.

```bash
curl.exe "http://127.0.0.1:8000/documents/" -H "Authorization: Bearer <access-token>"
```

## 6. Sharing endpoints

Only the owner can inspect or change a document's explicit read grants.

| Method and path | Purpose | Authentication |
| --- | --- | --- |
| `POST /documents/{document_id}/shares` | Grant read access to an existing user | Bearer + owner |
| `GET /documents/{document_id}/shares` | List explicit read grants | Bearer + owner |
| `DELETE /documents/{document_id}/shares/{user_id}` | Revoke an explicit read grant | Bearer + owner |

```bash
curl.exe -X POST "http://127.0.0.1:8000/documents/<document-id>/shares" -H "Authorization: Bearer <access-token>" -H "Content-Type: application/json" -d "{\"user_id\":\"00000000-0000-0000-0000-000000000202\"}"
```

## 7. Search endpoints

All search endpoints resolve readable document IDs before retrieval and never call
the generation LLM.

| Method and path | Purpose |
| --- | --- |
| `POST /search` | Semantic vector retrieval |
| `POST /search/hybrid` | Collect vector and BM25 candidates without fusion |
| `POST /search/hybrid/fused` | Combine vector and BM25 rankings with Reciprocal Rank Fusion |

There is no standalone keyword-search or reranker endpoint. Keyword and reranked
retrieval are available through the chat endpoint's `retrieval_mode`.

```bash
curl.exe -X POST "http://127.0.0.1:8000/search/hybrid/fused" -H "Authorization: Bearer <access-token>" -H "Content-Type: application/json" -d "{\"query\":\"HR-2026 leave policy\",\"top_k\":5,\"candidate_depth\":20,\"rrf_k\":60}"
```

## 8. RAG chat

`POST /chat/` accepts a `question` (1-4000 characters), `top_k` (1-10),
`retrieval_mode` (`keyword`, `vector`, `hybrid`, or `rerank`), and optional
non-negative `min_score`.

The runtime flow is:

```text
Bearer JWT
-> current PostgreSQL user
-> owned/shared document IDs
-> permission-aware retrieval
-> context construction
-> LLM or local fallback
-> answer + citations + retrieved contexts
```

The response reports `answer_mode` as `llm`, `local_fallback`, or `no_context`.
The current public Render service does not run a colocated Ollama model; when an
LLM provider is unavailable, the application may return HTTP `200` with
`answer_mode: "local_fallback"`. That is distinct from a `503` caused by an
unavailable authorization or retrieval dependency.

```bash
curl.exe -X POST "http://127.0.0.1:8000/chat/" -H "Authorization: Bearer <access-token>" -H "Content-Type: application/json" -d "{\"question\":\"How many annual leave days are available?\",\"top_k\":3,\"retrieval_mode\":\"hybrid\",\"min_score\":0.2}"
```

## 9. Typical workflow

1. Register two synthetic users.
2. Log in as the first user and save the returned token outside source control.
3. Upload a document; the uploader becomes its owner.
4. List, preview, search, or chat over that owner's authorized documents.
5. Grant the second user's UUID read access.
6. Log in as the second user and verify that the shared document is visible.
7. Revoke the grant and verify that the same second-user token is denied on its next request.

## 10. Deployment limitations

- Render Free may cold-start and its local filesystem is ephemeral.
- Neon PostgreSQL persists users, document metadata, ownership, and ACLs.
- Uploaded files and local retrieval indexes are not durable across Render instance replacement.
- The public service does not require or expose production credentials in this guide.
- API documentation describes current behavior; it is not a production availability or durability guarantee.
