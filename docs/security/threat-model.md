# Enterprise RAG System Threat Model

| Field | Value |
|---|---|
| Threat Model Version | `2.0` |
| Review Date | 2026-08-16 |
| System Scope | Current authenticated Enterprise RAG reference implementation |
| Evidence Basis | Current architecture and API documentation, committed tests, and frozen retrieval, generation, authorization, and security artifacts |
| Related Documents | [Architecture](../architecture.md), [Layered Defenses](layered-defenses.md), [Evaluation Results](../evaluation-results.md), [Failure Analysis](../failure-analysis.md) |

> **Security claims boundary:** This document models the current repository and records
> architecture-supported risks, existing controls, observed results, and residual gaps.
> It is not a compliance assessment, penetration-test certificate, production-security
> guarantee, or claim that prompt injection has been solved.

## 1. Scope

### In Scope

- Public FastAPI system, authentication, document, sharing, search, and chat routes.
- JWT authentication and current-user resolution from PostgreSQL.
- Document ownership, explicit read ACLs, revocation, and permission-aware retrieval.
- Document upload, local persistence, extraction, chunking, metadata, embeddings, and indexing.
- Keyword, Vector, Hybrid, Reciprocal Rank Fusion, heuristic reranking, and Cross-Encoder reranking paths.
- Context expansion, prompt construction, OpenAI-compatible LLM invocation, fallback behavior, output validation, citations, and returned contexts.
- Structured RAG/security logging and offline retrieval, generation, authorization, and prompt-injection evaluation artifacts.
- Local filesystem, PostgreSQL, optional Qdrant, local Ollama, and configured remote-provider boundaries.
- RAG-specific threats, authorization failures, citation integrity, and logging/artifact exposure.

### Out of Scope

- Generic host compromise, administrator compromise, arbitrary database administration, or repository write access.
- Full web-application review of SQL injection, XSS, CSRF, generic denial of service, dependency supply-chain compromise, and container-host hardening.
- Security properties of Render, Neon, Ollama, Qdrant, or another provider beyond the repository's integration boundary.
- Production availability, horizontal scaling, tenant administration, write/edit permissions, field-level policies, or regulatory compliance.
- Claiming that synthetic security cases predict unseen attacks or production adversaries.

### Current-System Boundary

The current system is authenticated and authorization-aware:

```text
Bearer JWT
-> current PostgreSQL user
-> ownership + active document ACLs
-> readable document-ID set
-> retrieval restricted to that set
-> context construction
-> LLM invocation
-> validated response
```

Owners have implicit read access. Non-owners require an explicit active read grant.
Everyone else is denied by default. The readable set is resolved on every request, so
revocation takes effect on the next request without replacing the JWT.

Authorization is deterministic application logic and occurs before document content
reaches final context or the LLM. Prompt instructions and output validation are
additional controls; they are not substitutes for authorization.

## 2. System Overview

The repository implements one FastAPI backend with:

- public registration, login, health, and discovery routes;
- Bearer-protected document, sharing, search, and chat routes;
- PostgreSQL-backed users, password hashes, document metadata, ownership, and ACLs;
- local uploaded files, extracted text, chunk JSON, vector indexes, and JSONL logs;
- optional Qdrant vector search;
- one OpenAI-compatible LLM client, normally using local Ollama for development and formal experiments;
- layered prompt/context/output controls;
- frozen offline evaluation artifacts.

The public chat schema exposes `keyword`, `vector`, `hybrid`, and `rerank`. The public
`rerank` mode uses the explainable heuristic reranker. A separately configured
`hybrid_rerank` path uses Vector + BM25 -> RRF -> Cross-Encoder and is used by formal
evaluation/security runners; it is not a public `ChatRequest` enum.

Development and formal model roles are intentionally separate:

```text
Development and smoke model: gemma3:4b
Formal evaluation reference: qwen3:8b
```

The public Render Free deployment persists relational state in Neon PostgreSQL, but its
local uploaded files, extracted text, indexes, and logs are ephemeral. Render Free does
not run a colocated Ollama service, so the local qwen security results are not cloud LLM
acceptance evidence.

## 3. Security Data Flows

### Legend

- `[TB-n]` is a trust boundary.
- `[U]` is attacker-influenceable or otherwise untrusted data.
- `[T]` is trusted application code or configuration.
- `[A]` is authenticated identity/authorization state derived from trusted application logic and PostgreSQL.
- Parsed, indexed, retrieved, or highly ranked document content remains untrusted data.

### Flow A — Authentication and Current-User Resolution

```text
[U] username + password
        |
      [TB-1]
        v
POST /auth/register or /auth/login
        |
        +-> normalized username
        +-> password hash verification / creation
        +-> PostgreSQL user record
        +-> signed JWT with subject and expiry

[U] Authorization: Bearer <token>
        |
      [TB-1]
        v
JWT signature / expiry validation
        |
        +-> subject extraction
        +-> current PostgreSQL user reload
        v
[A] UserIdentity
```

A valid JWT establishes identity, not document permission. Disabled, deleted, stale, or
otherwise invalid user state is rejected when the current user is reloaded.

### Flow B — Authenticated Document Ingestion

```text
[U] authenticated caller + filename + TXT/MD/PDF/DOCX bytes
        |
      [TB-1]
        v
POST /documents/upload
        |
        +-> current UserIdentity
        +-> filename/suffix/size/non-empty validation
        +-> UUID document identity
        v
      [TB-2]
        |
        +-> storage/uploads/{document_id}_{filename}
        +-> TXT/MD/PDF/DOCX parser
        +-> storage/texts/{document_id}.txt
        +-> page-aware sections and overlapping chunks
        +-> storage/index/chunks.json
        +-> embeddings and configured vector index
        +-> PostgreSQL DocumentRecord(owner_id=current user)
        v
[U] searchable content owned by the uploader
```

File-shape validation establishes processability, not author trust, factual correctness,
or instruction safety. Ownership limits who can retrieve the document, but an authorized
owner can still upload malicious or false content into that owner's accessible corpus.

Local file/index writes and PostgreSQL registration are not one atomic transaction.
Failure after a partial write may require explicit reconciliation.

### Flow C — Sharing and Revocation

```text
[A] current user + document ID + target user ID
        |
      [TB-3]
        v
owner check in PostgreSQL
        |
        +-> create/list/delete explicit read ACL
        v
current readable document set on the next request
```

Only the owner may manage explicit shares. A guessed document ID does not bypass
preview, chunk, search, or chat authorization.

### Flow D — Permission-Aware Search and RAG

```text
[U] Bearer token + question/query + retrieval parameters
        |
      [TB-1]
        v
current UserIdentity
        |
      [TB-3]
        v
PostgreSQL ownership + active ACL query
        |
        v
[A] frozenset of readable document IDs
        |
      [TB-4]
        v
retrieval restricted to authorized chunks
        |
        +-> Keyword / Vector / Hybrid / RRF / reranker
        +-> authorized adjacent expansion
        +-> final authorized contexts
        |
      [TB-5]
        v
[T] layered application instructions
[U] user question
[U] authorized but untrusted document text
        |
      [TB-6]
        v
configured OpenAI-compatible LLM provider
        |
      [TB-7]
        v
[U] model answer
        |
        +-> deterministic output/citation validation
        +-> response mapping
        +-> content-minimized security log
        v
answer + citations + contexts
```

Authorization is rechecked at useful stages: document listing, protected reads, vector
filtering, BM25 corpus construction, hybrid normalization, reranking, adjacent expansion,
and final context construction.

### Prompt Construction Flow

```text
Trusted application system message
        +
Untrusted user-question block
        +
Untrusted retrieved-document blocks
        +
source/chunk/page metadata and stable delimiters
        v
configured LLM
```

The layered policy defines the application message as instruction authority, the user
question as a task selector rather than a policy override, and retrieved documents as
evidence rather than instructions. This is a model-behavior control, not a hard trust
boundary.

### Citation Flow

```text
final authorized contexts
-> de-duplicate by chunk_id
-> validate citation membership
-> copy authorized document/chunk provenance
-> return citations with answer
```

Citations are mechanically derived from final contexts rather than parsed from answer
claims. Membership validation prevents citations to chunks outside final context, but
claim-level support remains unverified. An answer can still cite unrelated extra context.

### Logging and Evaluation Flow

```text
request/retrieval/security state
-> allow-listed JSONL event
-> local logs/rag.jsonl

offline fixed datasets + manifests
-> evaluation runner
-> full machine-readable artifacts under evals/results
```

Runtime logging excludes raw questions, prompt messages, document/context content,
answer text, credentials, authorization headers, protected canaries, and private
exception details. Logs still contain document/chunk IDs, rankings, model identity,
timing, usage, statuses, and security decisions.

Offline artifacts intentionally contain richer questions, answers, evidence, hashes, and
classifications. They are reproducibility assets and require their own access and
publication review.

## 4. Core Security Principles

```text
Authentication establishes identity.
Authorization establishes document access.
Retrieved Content = Data != Trusted Instruction.
Model Output = Untrusted Until Validated.
```

No retrieval score, parser success, embedding, reranker result, citation, or security
signal promotes document text to trusted instruction authority.

No prompt or model instruction may grant access to a document that the deterministic
ownership/ACL layer denied.

## 5. Assets

| Asset | Current concrete form | Security property |
|---|---|---|
| User identity | PostgreSQL user record and JWT subject | Integrity, availability |
| Password material | Argon2 password hashes | Confidentiality, integrity |
| JWT signing configuration | `JWT_SECRET_KEY`, token lifetime and algorithm settings | Confidentiality, integrity |
| Authorization state | Document owner IDs and explicit read ACLs | Integrity, availability |
| Original documents | `storage/uploads/` | Confidentiality, integrity, availability |
| Extracted text | `storage/texts/` | Confidentiality, integrity |
| Chunk corpus | `storage/index/chunks.json` and provenance metadata | Confidentiality, integrity, provenance |
| Search representations | Embeddings, FAISS/NumPy/JSON artifacts, optional Qdrant | Integrity, confidentiality, availability |
| Retrieval integrity | Authorized candidate identity, ordering, scores, cutoff | Integrity |
| Application instructions | Layered prompt, context format, grounding/citation rules | Integrity and precedence; secrecy is not an authorization boundary |
| Generation integrity | Model follows application policy rather than attacker content | Integrity |
| Citation integrity | Sources belong to authorized final context and support claims | Integrity, confidentiality |
| Configuration and credentials | `.env`, database/provider URLs, API keys, model settings | Confidentiality, integrity |
| Runtime logs | Request IDs, document/chunk IDs, rankings, timing, usage, security metadata | Confidentiality, integrity |
| Evaluation artifacts | Cases, outputs, evidence, model identities, metrics, reviews | Integrity, confidentiality, reproducibility |

The repository's frozen evaluation corpus is synthetic and small. This does not imply
that future runtime uploads are non-sensitive. Every user-uploaded document must be
handled according to its authorization and provider-exposure path.

## 6. Actors and Attacker Capabilities

### Actors

| Actor | Current role and trust |
|---|---|
| Unauthenticated caller | Can use public discovery, health, registration, and login routes; untrusted |
| Authenticated normal user | Can upload owned content and access owned/shared documents; inputs remain untrusted |
| Malicious authenticated user | Crafts requests, uploads malicious content to owned scope, probes authorization and model behavior |
| Document owner | Controls sharing for owned documents; trusted only for ownership decision, not content truth |
| Shared reader | Has explicit read access but cannot manage owner-only shares |
| Malicious external document author | Creates content later uploaded by an authorized user; may not control the API |
| Application owner/developer | Controls source, environment, deployment, migrations, and artifact publication |
| LLM provider/runtime | Receives finalized question/context messages and returns untrusted output |
| Embedding/vector provider | Receives text, embeddings, metadata, or filters when configured externally |
| Host/log/artifact reader | May access diagnostic or evaluation data outside normal API authorization |

### Realistic Application-Level Capabilities

An attacker may:

- register and authenticate under an attacker-controlled account;
- send requests that pass Pydantic shape/range validation;
- choose exposed search/chat retrieval parameters;
- upload supported documents up to the configured limit into attacker-owned scope;
- craft query or document text to influence lexical, vector, fusion, and reranker behavior;
- request sharing from or socially engineer a legitimate owner;
- probe document IDs, authorization failures, revocation timing, and source metadata;
- attempt direct prompt injection, indirect prompt injection, prompt leakage, context poisoning, or output-format manipulation;
- observe authorized answers, citations, contexts, safe errors, and timing.

Out-of-scope attacker capabilities include valid owner credentials obtained by host
compromise, JWT signing-key theft, server shell access, direct database/file writes, or
provider compromise. The consequences of those events are still relevant operationally,
but they are not modeled as normal API attacks here.

## 7. Trust Assumptions

- Application source, migrations, loaded configuration, and PostgreSQL authorization queries express intended behavior.
- JWT signing material remains secret and sufficiently strong.
- PostgreSQL, host filesystem, process, and local model files are protected against direct out-of-scope tampering.
- User questions, filenames, upload bytes, extracted text, metadata, and model outputs are untrusted.
- A valid JWT does not prove document access; the current owner/ACL lookup is authoritative.
- Stored document content remains untrusted as instruction even when the uploader owns it.
- The configured LLM provider can see every finalized question/context message sent to it.
- Local Ollama reduces third-party transport exposure but does not make generation trustworthy.
- Remote LLM or embedding configuration creates an external data-processing boundary.
- Optional Qdrant must enforce the provided authorized document-ID filter correctly.
- Frozen artifacts are evidence of specific runs, not universal security proof.

## 8. Trust Boundaries

| ID | Boundary | Data crossing | Security significance |
|---|---|---|---|
| TB-1 | External caller -> FastAPI | Credentials, JWT, query parameters, upload bytes | Validates request and identity; all caller data remains untrusted |
| TB-2 | Upload API -> parser/local storage | Untrusted bytes, filename, extracted text | Processable content may still be malicious or false |
| TB-3 | Application -> PostgreSQL identity/ACL state | User ID, document ID, ownership/share operations | Hard identity and authorization authority |
| TB-4 | Authorized corpus -> retrieval/reranker | Authorized chunk text, embeddings, metadata, scores | Determines which permitted but untrusted evidence reaches context |
| TB-5 | Retrieved content -> prompt/context | User question and authorized document text | Untrusted data shares an LLM request with trusted application instructions |
| TB-6 | Application -> configured LLM provider | System message, question, authorized contexts, generation settings | Provider receives prompt data; may be local or remote |
| TB-7 | LLM output -> application/user | Generated answer and usage metadata | Output may be manipulated, unsupported, or disclosing |
| TB-8 | Runtime/evaluation -> logs/artifacts | IDs, scores, timing, model usage, full offline evidence | Durable diagnostic data requires publication/access control |
| TB-9 | Application -> optional external vector/embedding service | Text, embeddings, metadata, authorized filters | External confidentiality and filter-enforcement dependency |
| TB-10 | GitHub/CI -> Render/Neon deployment | Source, image, migration secrets, database connections | Deployment integrity and secret-management boundary |

## 9. Attack Surface Inventory

| Surface | Authentication/authorization | Attacker influence | Returned/readable data | Persistent | Can reach LLM |
|---|---|---|---|---|---|
| `POST /auth/register` | Public | Username/password | User/token workflow status | PostgreSQL | No |
| `POST /auth/login` | Public | Credentials | JWT on success | Token lifetime | No |
| `GET /auth/me` | Bearer | Token | Current user identity | No | No |
| `POST /documents/upload` | Bearer; owner assigned | Filename and file content | Document ID, preview, paths/count metadata | Local files/index + PostgreSQL | After authorized retrieval |
| `GET /documents/` | Bearer; readable set | No global filter control | Only owned/shared metadata | No | No |
| Preview/chunks routes | Bearer + read access | Chooses document ID | Authorized text/chunks/metadata | No | No |
| Share routes | Bearer + owner | Target user/grant/revoke | Explicit ACL metadata | PostgreSQL | Indirectly |
| Search routes | Bearer + readable set | Query and parameters | Authorized raw result content/metadata | No | No generation |
| `POST /chat/` | Bearer + readable set | Question and retrieval parameters | Answer, authorized citations/contexts | Per request/log metadata | Yes |
| Parser/chunker | Owner-controlled document | Text/structure | Indirect through protected reads/search/chat | Yes | Yes |
| Retrieval/RRF/reranker | Query and authorized corpus | Ranking pressure | Scores/content through authorized endpoints | Derived | Selects context |
| Prompt builder | Query and authorized retrieved content | Injection attempts | Potentially via model output | No | Is LLM input |
| LLM provider | Receives finalized messages | Provider controls output | Provider-dependent | Provider-dependent | N/A |
| Citation builder | Final contexts | Corpus/query influence | Authorized source metadata | Per response | No |
| Structured log | Runtime state | Indirect | Host/log access only | Local JSONL | No |
| Evaluation artifacts | Fixed cases and outputs | Developer-controlled | Repository/artifact access | Yes | Offline only |
| Configuration/secrets | Operator-controlled | Out-of-scope direct attacker | Must not be API-visible | Yes | Selects external boundaries |

## 10. Existing Security Controls

`Existing` means implementation is present. It does not mean resistance is complete or
universally proven.

| Control | Status | Evidence and limitation |
|---|---|---|
| Request shape/range validation | Existing | Pydantic constrains types, lengths, modes, and numeric ranges; semantic intent remains untrusted |
| Password hashing and login verification | Existing | PostgreSQL users and Argon2 hashes; does not prevent credential reuse/phishing |
| Signed expiring JWT | Existing | Establishes identity; token theft remains possible and JWT alone grants no document access |
| Current-user reload | Existing | Rejects stale/invalid user state at protected routes |
| Owner and explicit read ACL | Existing | Deny by default; document-level read only, not group/tenant/field/write policy |
| Immediate revocation semantics | Existing | Readable set resolved per request; in-flight requests and already disclosed data cannot be recalled |
| Permission-aware document reads | Existing | List, preview, chunks, and share management enforce ownership/ACL rules |
| Permission-aware retrieval | Existing | Vector, BM25, hybrid, reranking, and context expansion operate on authorized document IDs |
| Upload validation | Existing | Extension, size, empty-content, parser, and filename checks; not content-trust validation |
| UUID document identity | Existing | Reduces collisions/guessability; secrecy of IDs is not authorization |
| Grounded layered prompt | Existing | Defines trusted instruction hierarchy; remains probabilistic model-visible text |
| Untrusted query/context framing | Existing | Stable delimiters and trust labels; not sanitization or authorization |
| Instruction-like context signal | Observe only | Produces metadata without removing content or changing rank; false positives/negatives remain |
| Deterministic output validation | Existing | Blocks defined canary/exact-clause/shape/citation-membership/fallback violations; not semantic truth validation |
| Citation membership validation | Existing | Prevents citation IDs outside final context; not claim-level attribution |
| Safe public errors | Existing | Reduces private exception/provider detail exposure |
| Content-minimized runtime logging | Existing | Excludes raw prompt/content/answer/secrets; metadata remains sensitive |
| `.gitignore` for runtime secrets/data | Existing | Reduces accidental commits; not secret rotation or runtime access control |
| CI and migration gates | Existing | Clean-runner tests and explicit database migrations; not a substitute for production hardening |
| Trusted-source approval/quarantine | Not implemented | Any authenticated owner may index content into their own accessible corpus |
| Claim-level groundedness/citation validation | Not implemented | Plausible false claims and unrelated extra citations may pass |
| Log retention/rotation/application ACL | Not implemented | Governed by host/deployment access; local log is ephemeral on Render Free |
| Durable cloud document storage | Not implemented | Render Free local artifacts disappear on restart/redeploy |
| Production rate limiting/MFA/session revocation | Not established by reviewed documents | Must not be inferred from JWT support |

## 11. Threat Summary and Risk Method

Ratings are qualitative engineering priorities. `Likelihood` reflects current
application prerequisites and reachable paths. `Impact` assumes runtime documents may be
private even though the frozen evaluation corpus is synthetic.

| Threat ID | Threat | Likelihood | Impact | Current risk | Evidence status |
|---|---|---|---|---|---|
| AUTH-001 | Broken authentication or token misuse | Medium | High | High | Architecture/tests establish controls; adversarial token testing not summarized here |
| ACL-001 | Authorization bypass or filter propagation failure | Low-Medium | Critical | High | Current implementation/tests support pre-LLM filtering; no external audit |
| DPI-001 | Direct Prompt Injection | High | Medium-High | High | Observed on frozen direct suite; reduced, not eliminated |
| IPI-001 | Indirect Prompt Injection | Medium | High | High | Observed on delivered malicious-document suite; no paired improvement |
| MD-001 | Malicious Document Persistence | Medium | High | High | Authenticated owner path exists; scope is owner/shared readers rather than global corpus |
| SPL-001 | System Prompt Leakage | Medium | Medium | Medium | Exact leakage reduced on known suite; paraphrased leakage coverage limited |
| SIL-001 | Sensitive Information Leakage | Low-Medium through API; provider/host dependent | Critical | High | ACL reduces cross-user exposure; provider/log/artifact and semantic leakage remain |
| CP-001 | Context Poisoning | Medium | High | High | Plausible false evidence remains successful in known evaluation |
| CI-001 | Citation Integrity Manipulation | Medium | Medium-High | Medium-High | Membership checks exist; claim-level support absent |
| LOG-001 | Logging or Artifact Exposure | Low for API caller; host/repository dependent | High | Medium | Runtime content minimized; offline artifacts richer |
| STO-001 | Cross-store inconsistency or ephemeral artifact loss | Medium | Medium | Medium | Known non-atomic and Render durability limitations |

The frozen prompt-injection evaluation predates authentication/ACL implementation. Its
ASR values measure prompt/context/output behavior under that frozen system configuration;
they do not test current cross-user isolation. Current authorization is assessed
separately through architecture and permission tests.

## 12. Detailed Threat Records

### AUTH-001 — Broken Authentication or Token Misuse

- **Threat:** An attacker obtains or forges a valid identity through credential compromise, weak token configuration, token theft, or implementation error.
- **Entry points:** Registration, login, Bearer token processing, environment configuration.
- **Target assets:** User identity, owned/shared document access, sharing authority.
- **Trust boundaries:** TB-1, TB-3.
- **Current controls:** Password hashing, signed expiring JWT, current-user reload, missing/invalid/expired/stale credential rejection, environment-secret requirements.
- **Residual risk:** Stolen valid credentials or signing keys bypass normal identity checks. MFA, refresh-token management, token revocation lists, rate limits, and account recovery are not established by the reviewed documents.
- **Required verification:** Authentication negative tests, secret-strength checks, expiry/stale-user tests, deployment-secret review.

### ACL-001 — Authorization Bypass or Filter Propagation Failure

- **Threat:** A user reads, searches, cites, or sends another user's document to the LLM without ownership or an active grant.
- **Entry points:** Document IDs, list/preview/chunks, sharing endpoints, vector/BM25/hybrid/rerank/context paths.
- **Attacker:** Authenticated non-owner.
- **Target assets:** Document content, existence, metadata, derived embeddings, citations.
- **Trust boundaries:** TB-3, TB-4, TB-5.
- **Current controls:** Owner implicit access, explicit read ACL, deny-by-default decisions, per-request readable set, vector filters, authorized BM25 subset, rechecks during hybrid/rerank/adjacent expansion, protected document reads.
- **Control status:** Implemented and tested within the repository; not independently audited.
- **Residual risk:** A missed code path, incorrect optional-backend filter, stale cache added later, or metadata propagation bug could reintroduce exposure. Already returned data cannot be revoked retroactively.
- **Required verification:** Cross-user negative tests for every endpoint and retrieval mode, immediate-revocation tests, optional Qdrant filter tests, citation/context non-disclosure assertions.

### DPI-001 — Direct Prompt Injection

- **Threat:** A user places instructions in the question to compete with application policy.
- **Entry point:** `POST /chat/` question.
- **Preconditions:** Authenticated request and a path that invokes the configured model.
- **Target assets:** Generation integrity, prompt confidentiality, groundedness, authorized context.
- **Trust boundaries:** TB-1, TB-5, TB-6, TB-7.
- **Current controls:** Layered instruction hierarchy, untrusted question framing, grounding/abstention rules, deterministic leakage/shape validation, content-minimized security logging.
- **Observed evidence:** On the paired known direct suite, final-response ASR fell from 30.0% to 10.0%. One direct output-format attack remained successful and one lower-severity partial regression remained.
- **Residual risk:** High. Encodings, paraphrases, unseen objectives, and semantically unsafe but schema-valid outputs may bypass deterministic rules.

### IPI-001 — Indirect Prompt Injection

- **Threat:** Retrieved document text contains instructions that the LLM may treat as authority.
- **Entry point:** Authenticated document upload or an authorized user ingesting externally authored material.
- **Preconditions:** Content is parsed, indexed, authorized for the requester, retrieved, selected, and sent to the model.
- **Target assets:** Generation integrity, prompt confidentiality, answer correctness, authorized context.
- **Trust boundaries:** TB-2, TB-4, TB-5, TB-6, TB-7.
- **Current controls:** Document ownership/ACL scope, explicit untrusted-document rule, stable framing, observe-only signal, output validation.
- **Observed evidence:** All frozen malicious chunks reached context in the historical suite. On the valid-in-both paired set, indirect ASR remained 12.5% in both baseline and layered modes.
- **Residual risk:** High. Authorization limits who is exposed but does not make an authorized malicious document safe.

### MD-001 — Malicious Document Persistence

- **Threat:** A supported document persists malicious instructions, false facts, retrieval-manipulating text, or misleading metadata.
- **Entry point:** `POST /documents/upload`.
- **Attacker:** Authenticated owner, compromised account, or malicious external author whose file is uploaded by a legitimate user.
- **Target assets:** Corpus, retrieval, answer, and citation integrity.
- **Trust boundaries:** TB-1, TB-2, TB-4.
- **Current controls:** Authentication, owner assignment, extension/size/parser checks, UUID IDs, provenance metadata, ACL-limited exposure.
- **Control status:** Processing and authorization controls exist; trusted-source approval, quarantine, signer verification, and factual validation do not.
- **Residual risk:** High within the owner/shared-reader scope. A malicious document can repeatedly affect future authorized queries until removed or storage is reconciled.

### SPL-001 — System Prompt Leakage

- **Threat:** Query or document content induces the model to reveal application instructions or prompt structure.
- **Target assets:** Layered system message, prompt format, grounding/citation rules.
- **Current controls:** Prompt contains no credentials; anti-disclosure instruction; exact-clause and test-canary validation; logs exclude prompt text.
- **Observed evidence:** Direct exact-clause leakage success fell from 1/2 to 0/2 on the frozen comparison, with one partial remaining. Indirect extraction remained 0/1 in both modes.
- **Residual risk:** Medium. Paraphrased or semantic disclosure is only partially covered. Prompt secrecy must never protect credentials or authorization.

### SIL-001 — Sensitive Information Leakage

- **Threat:** Document content, source metadata, internal instructions, operational data, or evaluation evidence reaches an unauthorized or unintended recipient.
- **Entry points:** Protected document/search/chat responses, configured LLM/embedding/vector providers, logs, artifacts, deployment secrets.
- **Target assets:** Documents, filenames, existence, context, prompt, credentials, usage patterns.
- **Current controls:** JWT identity, owner/ACL filtering, pre-LLM permission-aware retrieval, authorized citation/context construction, safe errors, content-minimized runtime logs, ignored runtime secret/data files.
- **Residual risk:** Critical if an authorization path fails; otherwise remains through authorized users, remote-provider processing, host access, artifact publication, semantic model leakage, and accidental secret commits.
- **Evidence boundary:** Frozen prompt-injection ASR is not a cross-user ACL test. Current permission tests and architecture establish the intended deterministic boundary.

### CP-001 — Context Poisoning

- **Threat:** Manipulated but authorized evidence reaches final context and produces a plausible false or biased answer.
- **Entry point:** Uploaded corpus plus query-dependent retrieval/ranking.
- **Current controls:** Authorized corpus restriction, multiple retrieval signals, stable RRF, optional reranking, untrusted framing, grounded prompt, limited exact-evidence/no-context behavior.
- **Observed evidence:** A known plausible false-evidence case remained successful because it did not violate the deterministic leakage/contract validator.
- **Residual risk:** High. False evidence does not need to look like an instruction and cannot be solved by ACLs or delimiter framing.

### CI-001 — Citation Integrity Manipulation

- **Threat:** Malicious or unrelated context appears authoritative through returned citations or exposes source metadata unnecessarily.
- **Current controls:** Authorized final-context source set, stable citation schema, deduplication, citation-membership validation, system instruction requesting direct support.
- **Observed evidence:** Frozen generation evaluation found acceptable answers with unrelated extra-document citations in five cases. No aggregate citation regression occurred between baseline and layered prompt modes.
- **Residual risk:** Medium-High. No claim-to-source entailment validator exists.

### LOG-001 — Logging and Artifact Exposure

- **Threat:** A host, repository, or artifact reader learns document identities, access patterns, questions, answers, attack payloads, or model behavior.
- **Current controls:** Runtime allow-list excludes raw content and secrets; `.gitignore` excludes runtime logs/storage; stable artifact provenance enables review.
- **Residual risk:** Runtime metadata remains sensitive. Offline artifacts contain substantially richer evidence. Retention, rotation, encryption, and application-level log access policy are not implemented.

### STO-001 — Cross-Store Inconsistency and Ephemeral Loss

- **Threat:** PostgreSQL metadata and local file/index artifacts diverge, or Render restarts erase local RAG data while relational records remain.
- **Current controls:** Explicit error handling, documented filesystem contract, migrations for relational state.
- **Residual risk:** Partial artifacts, stale document metadata, unavailable previews/search, and manual reconciliation. This is primarily integrity/availability risk but may affect authorization reasoning if future cleanup paths are incomplete.

## 13. Direct, Indirect, Malicious-Document, and Poisoning Distinctions

```text
Direct Prompt Injection
= attacker instruction arrives in the current user question.

Indirect Prompt Injection
= attacker instruction arrives through retrieved document content.

Malicious Document
= persistent attacker-controlled content or delivery vehicle.

Context Poisoning
= manipulated evidence reaches final context and alters the answer.
```

A malicious document does not affect a request unless the requester is authorized to
access it and retrieval/context selection delivers it. Authorization prevents cross-user
exposure; it does not establish factual or instructional trust inside the authorized
corpus.

## 14. Threat-to-Control and Evidence Traceability

| Threat | Primary current controls | Current evidence | Important gap |
|---|---|---|---|
| AUTH-001 | Password hashing, signed expiring JWT, current-user reload | Authentication tests and API behavior | MFA/rate limits/token revocation not established |
| ACL-001 | Owner/ACL checks and pre-LLM authorized retrieval | Permission and revocation tests | Independent audit and future-path regression risk |
| DPI-001 | Prompt hierarchy, framing, output validation | Paired direct ASR 30% -> 10% | Unseen and semantic attacks |
| IPI-001 | ACL scope, untrusted-document framing, signal, validator | Paired indirect ASR unchanged at 12.5% | False evidence and unseen instructions |
| MD-001 | Authentication, owner assignment, ACL-limited access | Ingestion and ownership behavior | No provenance approval/quarantine |
| SPL-001 | No secrets in prompt, anti-leak rule, exact-clause validation | Exact direct leakage success reduced | Paraphrased/semantic leakage |
| SIL-001 | Pre-LLM authorization, safe errors, minimized runtime log | Permission-aware retrieval tests | Provider/host/artifact exposure and semantic leakage |
| CP-001 | Authorized corpus, retrieval quality controls, framing | Known false-evidence residual case | Factual validation/claim grounding |
| CI-001 | Authorized context, membership validation, citation schema | Citation metrics and failure analysis | Claim-level support validation |
| LOG-001 | Runtime allow-list and ignored runtime files | Schema/review evidence | Retention/access/encryption policy |
| STO-001 | Documented contracts and error handling | Deployment/failure analysis | Atomicity and durable cloud artifacts |

Machine-readable historical artifact names may retain `W*` identifiers because changing
them would break provenance and hash-linked references. Those identifiers are experiment
IDs, not current project-development instructions.

## 15. Risk Priorities

1. **ACL-001 / SIL-001:** maintain complete authorization propagation before every content-returning or LLM-bound path.
2. **IPI-001 / CP-001 / MD-001:** authorized content can still be malicious or false; access control and content trust are separate problems.
3. **DPI-001 / SPL-001:** layered controls reduce selected known attacks but remain probabilistic and incomplete.
4. **CI-001:** citation presence and membership are not equivalent to claim-level support.
5. **AUTH-001:** identity controls depend on credential hygiene, signing-key protection, and deployment configuration.
6. **LOG-001 / STO-001:** diagnostic evidence and cross-store state require lifecycle and reconciliation controls.

## 16. Observability Security Value and Risk

### Security Value

- Request IDs support incident correlation.
- Retrieved/context/citation IDs help reconstruct what evidence was exposed.
- Security mode, policy, defense IDs, signal status, output-block decisions, model, usage, timing, and error stage support investigation.
- Content minimization reduces accidental raw-data logging.

### Security Risk

- Document/chunk IDs and rankings reveal corpus access patterns.
- Model and usage fields reveal operational behavior.
- Frozen evaluation artifacts contain questions, answers, malicious fixtures, evidence, and classifications.
- Local files have no repository-level guarantee of encryption, retention, rotation, or per-user access control.

Observability is both a security control and a data store requiring deliberate
publication and access review.

## 17. Residual Risks

- Prompt injection remains a probabilistic model-behavior risk.
- ACLs prevent unauthorized retrieval but do not make authorized documents truthful or safe.
- A compromised legitimate account can exercise all permissions of that user.
- Output validation may miss paraphrased leakage, subtle exfiltration, and semantic falsehoods.
- Retrieval and reranker changes can move malicious evidence into or out of context.
- Remote providers introduce processing, retention, and jurisdiction assumptions outside this repository.
- Optional Qdrant or future backends must preserve authorization filters exactly.
- Citation accuracy cannot be inferred from citation presence or context membership alone.
- Runtime logs and offline artifacts remain sensitive despite content minimization.
- Local storage and PostgreSQL metadata are not atomically committed.
- Render Free does not provide durable document/index storage or local Ollama generation.

## 18. What Is Proven

- Protected routes resolve a current authenticated PostgreSQL user from a Bearer JWT.
- Ownership and explicit read ACLs determine readable document IDs.
- Document reads and retrieval paths are designed to filter unauthorized documents before context and LLM invocation.
- Revocation is evaluated on the next request rather than encoded permanently in the JWT.
- Layered prompt/context/output/logging controls are implemented with stable defense identifiers.
- Frozen known-suite evidence shows lower direct final-response ASR and no paired indirect improvement.
- Current citation, evaluation, deployment, and storage limitations are explicitly documented.

## 19. What Is Not Proven

- No result establishes production security, compliance, tenant isolation beyond current document-level read ACLs, or prompt-injection immunity.
- Frozen prompt-injection results do not test current cross-user ACL isolation.
- No confidence interval or unseen-attack generalization is supported by the small synthetic suites.
- Semantic correctness, groundedness, claim-level citation support, and zero hallucination are not established.
- MFA, production rate limiting, session revocation, durable cloud document storage, and centralized log governance are not established.
- No control should be considered complete solely because it appears in this threat model.

## 20. Review Triggers

Update this threat model when any of the following changes:

- authentication, token, password, account-state, or secret-management behavior;
- document ownership, ACL schema, permission semantics, or revocation behavior;
- any document, search, reranking, context, citation, or LLM path that may bypass the readable-ID set;
- upload provenance, approval, quarantine, deletion, or reconciliation workflows;
- prompt, signal, output validator, citation validator, or security-policy versions;
- public response schemas or context/source metadata exposure;
- LLM, embedding, vector-store, database, or deployment provider boundaries;
- logging/artifact content, access, retention, encryption, or publication policy;
- security datasets, rubrics, models, or formal result artifacts;
- production exposure, tenant model, write permissions, or regulatory requirements.
