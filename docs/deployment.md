# Deployment Guide — Render Free + Neon Free

This guide documents the demo deployment profile for the Enterprise
RAG backend. It deliberately targets a zero-cost public API deployment, not a
production infrastructure claim.

## 1. Deployed Architecture

```text
GitHub main
    -> GitHub Actions CI
    -> Render Free Docker web service
        -> Neon Free PostgreSQL 18

Manual GitHub Actions migration
    -> direct Neon connection
    -> alembic upgrade head
```

The backend and database should both use their Singapore regions. The runtime
uses Neon's pooled connection endpoint. Alembic uses the direct endpoint.

## 2. Zero-Cost Guardrails

- Keep both accounts on their explicit Free plans.
- Do not add a payment method to Render or Neon.
- Do not add a Render persistent disk, custom domain, paid instance, or paid
  pipeline capacity.
- Stop if either dashboard displays a non-zero estimated cost or requires an
  upgrade to continue.
- Free-tier suspension is acceptable; an automatic paid fallback is not.

Expected recurring cost under this contract: **USD 0**.

## 3. What This Deployment Proves

The acceptance scope is intentionally limited to:

- the Docker image builds on the deployment platform;
- the FastAPI process starts as a non-root container user;
- the platform-provided HTTPS endpoint reaches `GET /health`;
- PostgreSQL 18 is reachable over TLS;
- explicit Alembic migrations reach the current head;
- registration and login persist data in Neon;
- GitHub CI gates automatic Render redeploys;
- logs and recent deployment rollback are available.

It does **not** prove:

- production availability or performance;
- persistent uploaded documents, extracted text, FAISS indexes, or JSONL logs;
- live Ollama inference with `gemma3:4b`;
- formal `qwen3:8b` deployment acceptance;
- end-to-end persistent RAG after a restart.

## 4. Filesystem Contract

Render Free has an ephemeral filesystem. The following runtime paths can be
written while an instance is alive but are lost on a restart, idle spin-down,
or redeploy:

```text
/app/storage/uploads
/app/storage/texts
/app/storage/index
/app/logs/rag.jsonl
```

User records, password hashes, document metadata, ownership, and ACL rows live
in Neon and are persistent. Do not use document upload as durable storage in
this deployment profile: preserving its PostgreSQL row would not preserve the
corresponding local file and index.

## 5. LLM Contract

The deployment preserves the repository's model roles:

```text
Development model: gemma3:4b
Formal evaluation/final model: qwen3:8b
```

Render Free does not run an Ollama service. `LLM_PROVIDER` remains `ollama`, and
the configured localhost endpoint is intentionally unreachable. The backend
must report its existing provider-unavailable behavior if an LLM route is
called. No fake response, third-party provider, or silent model substitution is
introduced.

## 6. Create the Neon Free Database

1. Sign in to Neon without adding a payment method.
2. Create a Free project in AWS Asia Pacific (Singapore).
3. Select PostgreSQL 18.
4. Keep scale-to-zero enabled.
5. From the Connect dialog, obtain both:
   - the direct connection string for migrations;
   - the pooled connection string for the Render runtime.
6. Preserve the TLS query parameters supplied by Neon.
7. Change only the SQLAlchemy driver prefix:

```text
postgresql://...
    ->
postgresql+psycopg://...
```

Never paste either real connection string into a tracked file, issue, report,
chat, or command output.

## 7. Configure the Migration Secret

In the GitHub repository, create this Actions secret:

```text
NEON_DIRECT_DATABASE_URL
```

Its value is the direct Neon URL using the `postgresql+psycopg://` prefix. It
must not use a hostname containing `-pooler`.

Run the **Deploy database migration** workflow manually from `main` and select
`MIGRATE`. The workflow performs:

```text
clean exact dependency install
-> validate direct endpoint
-> python -m alembic upgrade head
-> python -m alembic current --check-heads
-> verify PostgreSQL major version 18
```

Do not create the Render service until this workflow succeeds.

## 8. Create the Render Free Service

1. Sign in to Render without adding a payment method.
2. Choose **Blueprint** and connect this GitHub repository.
3. Use the repository-root `render.yaml`.
4. Confirm that the proposed service is:
   - type `web`;
   - runtime `docker`;
   - plan `free`;
   - region `singapore`;
   - one instance;
   - health check `/health`;
   - auto-deploy `checksPass`.
5. When prompted for `DATABASE_URL`, enter the pooled Neon URL using the
   `postgresql+psycopg://` prefix.
6. Let Render generate `JWT_SECRET_KEY`; do not replace it with a committed
   placeholder.
7. Confirm the displayed cost remains zero before applying the Blueprint.

`PORT=8000` intentionally matches the existing Dockerfile and Compose runtime.

## 9. Acceptance Checks

Record the deployed Git commit and verify:

```text
GET https://<render-service>.onrender.com/health
-> HTTP 200
-> {"status":"ok"}
```

Then use the generated OpenAPI UI only with disposable demo credentials:

```text
POST /auth/register
POST /auth/login
```

After one restart or redeploy, repeat login with the same disposable account.
This verifies database persistence independently of the process-only health
endpoint. Do not upload real documents or production credentials.

Also verify:

- the live Render deploy references the intended commit;
- runtime logs contain no secret values;
- a failed health check does not replace the last healthy deploy;
- the latest eligible Render rollback is visible;
- a request after idle spin-down eventually recovers through a cold start.

## 10. Redeploy and Rollback

Render waits for GitHub checks because `autoDeployTrigger` is `checksPass`.

```text
push to main
-> CI success
-> Render build
-> /health succeeds
-> new deploy receives traffic
```

If the build or health check fails, investigate Render build/runtime logs and
leave the previous successful deploy active. Application rollback does not
automatically downgrade the database. Database schema changes require their own
forward-compatible migration decision.

## 11. Failure Semantics

- Missing `DATABASE_URL`: database-backed routes fail closed.
- Missing/short `JWT_SECRET_KEY`: authentication fails closed.
- Neon scale-to-zero: the first database request can be slower.
- Render idle spin-down: the first public request can take about a minute.
- Neon unavailable: authentication and registry operations should return their
  existing service-unavailable responses.
- Ollama unavailable: LLM-backed routes follow the existing provider error
  path; this deployment does not claim successful generation.
- Render restart/redeploy: local document and log files are discarded.

## 12. Decommissioning

To decommission the demo deployment without leaving resources behind:

1. Delete the Render Blueprint/service.
2. Delete the Neon project only after confirming no demo data is needed.
3. Delete `NEON_DIRECT_DATABASE_URL` from GitHub Actions secrets.
4. Confirm that neither platform has a payment method or paid resource.
