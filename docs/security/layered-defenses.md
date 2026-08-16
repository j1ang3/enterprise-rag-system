# Layered Security Controls

## 1. Scope

This document describes the security controls implemented in the current Enterprise RAG
system. It covers:

- authenticated identity and document-level authorization;
- permission-aware retrieval before context construction;
- direct and indirect prompt-injection controls;
- trusted/untrusted message framing;
- deterministic output and citation validation;
- content-minimized security logging;
- frozen baseline-versus-layered evaluation evidence;
- residual risks that remain outside these controls.

The controls are layered because no single prompt, filter, detector, or authorization
check can address every failure mode. Deterministic identity and authorization controls
protect document access. Prompt, context, output, and logging controls address model and
response behavior after the authorized corpus has been established.

> **Claims boundary:** These controls reduce specific observed risks. They do not prove
> prompt-injection immunity, factual correctness, claim-level grounding, production
> security, or unseen-attack generalization.

## 2. Security Architecture

The current security-critical ordering is:

```text
Bearer JWT
-> current PostgreSQL user
-> ownership + active ACL evaluation
-> readable document-ID set
-> retrieval restricted to authorized chunks
-> reranking and context expansion within authorized documents
-> layered prompt and untrusted-data framing
-> configured LLM
-> deterministic output/citation validation
-> authorized response
-> content-minimized security event
```

The hard boundary is authorization before retrieval/context/LLM exposure. Prompt and
output layers do not grant access and cannot repair an authorization failure after
unauthorized content has already reached a model request.

## 3. Goals and Non-Goals

### Goals

- establish caller identity on every protected route;
- deny document access unless the caller is owner or an explicitly shared reader;
- restrict all retrieval and final context construction to readable documents;
- make application instructions and untrusted question/document data structurally distinct;
- preserve normal document content instead of using a keyword blacklist;
- block deterministic known output and citation-contract violations without echoing blocked text;
- record security decisions without logging raw prompts, documents, or answers;
- preserve frozen baseline/layered experiment provenance.

### Non-Goals

- proving that prompt injection is solved;
- treating authentication as authorization;
- treating an authorized document as trusted or factually correct;
- recognizing every malicious instruction, false fact, paraphrased leak, or encoded payload;
- filtering documents solely because they contain words such as `ignore`, `prompt`, `system`, or `instruction`;
- providing claim-level citation entailment or semantic-factual verification;
- implementing MFA, field-level permissions, group/tenant administration, write/edit ACLs, or a full security operations platform;
- converting historical prompt-injection metrics into evidence about current cross-user ACL isolation.

## 4. Control Layers

The system applies controls at distinct boundaries:

| Layer | Purpose | Primary failure modes |
|---|---|---|
| Identity | Establish the current user | Forged, expired, stale, or missing credentials |
| Authorization | Determine readable documents | Cross-user document access and source disclosure |
| Retrieval enforcement | Keep unauthorized chunks out of candidates and contexts | Filter propagation failures |
| Prompt hierarchy | Define trusted application instructions | Direct and indirect instruction conflict |
| Context framing | Mark user/document content as untrusted data | Document instructions interpreted as authority |
| Security signal | Observe instruction-like context | Investigation and measurement gaps |
| Output validation | Fail closed on deterministic violations | Exact leakage, invalid shape, unauthorized citation membership, unsafe fallback |
| Logging | Record decisions without raw content | Incident-analysis gaps and accidental sensitive logging |

Identity, authorization, and retrieval enforcement are deterministic application controls.
Prompt hierarchy and context framing influence a probabilistic model. Output validation
covers only conditions the application can check reliably.

## 5. Identity and Authentication Controls

Protected document, sharing, search, and chat routes require:

```text
Authorization: Bearer <access-token>
```

The authentication dependency:

1. validates token structure, signature, and expiry;
2. extracts the JWT subject;
3. reloads the current user from PostgreSQL;
4. rejects missing, invalid, expired, or stale credentials;
5. returns a trusted `UserIdentity` object to application services.

The JWT carries identity, not document permissions. Permission is never inferred from the
request body, filename, source metadata, prompt text, or model output.

### Limitations

- A stolen valid credential has the privileges of its user.
- MFA, refresh-token rotation, token revocation lists, login throttling, and account recovery are not established by this document.
- Signing-key strength and deployment-secret handling remain operational dependencies.

## 6. Ownership and ACL Authorization

The PostgreSQL authorization model provides:

- uploader ownership at document registration;
- implicit owner read access;
- explicit read grants for non-owners;
- owner-only share inspection, grant, and revoke operations;
- deny-by-default behavior for everyone else;
- per-request resolution of readable document IDs.

```text
current user
-> owned document IDs
   union
-> active explicit read-grant document IDs
-> readable document-ID set
```

Because permissions are queried on each request, revocation takes effect on the reader's
next request without replacing the JWT.

### Authorization Enforcement Points

- document listing filters in PostgreSQL rather than returning a global list;
- preview and chunk endpoints authorize before reading local content;
- sharing endpoints require ownership;
- vector retrieval receives allowed document IDs;
- optional Qdrant requests include a document-ID filter;
- BM25 is constructed from an authorized chunk subset;
- hybrid normalization, reranking, and adjacent expansion preserve or recheck document membership;
- final citations and contexts derive only from authorized final contexts.

### Limitations

- The policy is document-level read access, not field-, row-, group-, tenant-, write-, or edit-level authorization.
- Already returned content cannot be withdrawn after revocation.
- Every new retrieval backend or content-returning endpoint must preserve the same filter contract.

## 7. Security Modes and Historical Comparability

`RAG_SECURITY_MODE` is centralized in settings and accepts `baseline` or `layered`.
Normal application traffic defaults to `layered`.

### `baseline`

The baseline mode preserves the historical prompt/context/output behavior used by frozen
retrieval and security artifacts. It remains available for reproducibility and controlled
comparison; it is not the recommended normal application mode.

Historical policy identifier:

```text
w9-t2-t3-baseline.v1
```

### `layered`

The layered mode enables the five stable `DEF-*` controls described below.

Historical policy identifier:

```text
w9-t4-layered-defenses.v1
```

The `w9-*` strings are stable machine/provenance identifiers. Their presence does not
mean the current project is organized as a public weekly task log.

Switching prompt-security mode does not select a separate RAG implementation. Both modes
use the same configured retrieval, reranking, context, provider, model, generation, and
citation components for a controlled comparison.

Authentication and ACL enforcement are current application boundaries outside the
historical `DEF-*` prompt-defense registry. The frozen baseline/layered security
comparison predates those ACL controls and must not be interpreted as an authorization
test.

## 8. Stable Prompt/Output Defense Registry

| Defense ID | Layer | Primary threats | Action |
|---|---|---|---|
| `DEF-PROMPT-001` | Prompt | DPI-001, IPI-001, SPL-001 | Trusted application-instruction hierarchy |
| `DEF-CONTEXT-001` | Context | IPI-001, MD-001, SIL-001 | Full-text untrusted question/document framing |
| `DEF-SIGNAL-001` | Input observability | IPI-001, MD-001, LOG-001 | Observe-only instruction-like signal |
| `DEF-OUTPUT-001` | Output | SPL-001, SIL-001, CI-001 | Deterministic fail-closed validation |
| `DEF-LOG-001` | Observability | DPI-001, IPI-001, LOG-001 | Content-minimized security event fields |

The runtime registry in `app/security/defenses.py` is authoritative. The same IDs are
recorded in the frozen layered-defense configuration and comparison artifacts.

## 9. `DEF-PROMPT-001` — Trusted Instruction Hierarchy

The layered system message defines three trust levels:

1. application instructions in the system message;
2. the user request as a question selector, not a policy override;
3. retrieved documents as untrusted evidence, never instruction authority.

It directs the model not to execute role changes, hidden-instruction requests, context
copy requests, unsupported-fact requests, or output-policy changes found in untrusted
blocks. It retains grounding, abstention, concise-answer, and citation rules.

### Security Value

- makes instruction precedence explicit;
- distinguishes task content from application policy;
- reduces ambiguity when retrieved text resembles commands;
- provides a stable policy identity for evaluation.

### Limitations

The prompt is model-visible text. It is not a hard security boundary and cannot guarantee
obedience. Encoded, paraphrased, or novel attacks may still succeed.

## 10. `DEF-CONTEXT-001` — Untrusted Query and Document Framing

Layered messages wrap the user question and retrieved documents in separately labeled
untrusted blocks. Each document block includes source metadata, complete selected content,
a trust label, character count, and stable begin/end markers.

The system message states that delimiter-like strings found inside a document remain
data. The implementation does not delete or rewrite ordinary words associated with
prompt injection.

### Security Value

- preserves legitimate content, including security documentation;
- marks provenance and trust status explicitly;
- separates documents from application instructions;
- prevents delimiter text alone from creating trusted authority.

### Limitations

- Delimiters aid interpretation but do not sanitize content.
- Framing does not establish factual truth.
- Framing does not authorize a document; authorization has already occurred earlier.
- A plausible false statement may contain no instruction-like language.

## 11. `DEF-SIGNAL-001` — Observe-Only Instruction Signal

The lightweight detector requires both a general control verb and an authority/control
target before emitting:

```text
SEC-CTX-INSTRUCTION-LIKE
```

It returns metadata such as signal IDs, feature IDs, and flagged chunk IDs. Its action is
always `observe_only`:

- no chunk is removed or rewritten;
- no retrieval score or order changes;
- no authorization decision changes;
- absence of a signal does not make a document trusted;
- presence of a signal does not prove malicious intent.

### Security Value

- exposes potentially instruction-like context for logs and evaluation;
- supports incident reconstruction without recording raw content;
- avoids turning a brittle lexical detector into a blocking security boundary.

### Limitations

False negatives remain possible through paraphrase, encoding, indirect wording, or pure
false evidence. False positives remain possible in legitimate policy or security text.

## 12. `DEF-OUTPUT-001` — Deterministic Output Validation

Layered output validation checks conditions the application can verify:

- a test-side protected canary appears in output;
- multiple exact protected application-prompt clauses appear in output;
- answer/citation shape is invalid;
- a citation does not belong to the final authorized context set;
- a provider-error fallback would serialize retrieved context;
- the validator itself raises an unexpected error.

On a defined violation, the layer fails closed:

1. replace the answer with a generic non-echoing safe response;
2. clear citations when required;
3. record a stable blocked reason and defense ID;
4. exclude blocked content and protected canaries from metadata/logs.

Baseline mode preserves historical behavior and skips the new validator for controlled
comparison.

### Security Value

- enforces conditions that do not depend on subjective model judgment;
- prevents exact protected-text leakage and invalid source membership;
- avoids returning raw retrieved context through an unsafe local fallback;
- fails closed on unexpected validator failure in layered mode.

### Limitations

The validator cannot reliably detect:

- paraphrased or semantic prompt disclosure;
- plausible but false facts;
- subtle exfiltration;
- claim-level citation mismatch;
- every adversarial encoding;
- general groundedness or answer correctness.

## 13. Synthetic Canary Boundary

The production prompt contains no secret canary. Tests may pass a synthetic protected
canary directly to the security policy for output comparison. The value is not inserted
into production messages or logs.

Document markers used by indirect-attack fixtures are not automatically protected prompt
canaries. Reproduction of document text demonstrates document-content influence, not
necessarily trusted-prompt leakage.

No credential, API key, JWT secret, or authorization decision may depend on prompt-canary
secrecy.

## 14. Citation and Context Membership Controls

The response citation list is built from final context metadata rather than accepted from
arbitrary model output. Layered validation verifies that cited chunks belong to the final
context set, which has already been restricted to readable documents.

This provides two useful guarantees:

- an unauthorized document filtered before context cannot be introduced merely by a model-generated citation ID;
- a citation outside the finalized context set is a deterministic contract violation.

It does not prove that each cited source supports each answer claim. Frozen failure
analysis observed acceptable answers accompanied by unrelated extra-document citations.
Claim-level attribution remains a separate unsolved problem.

## 15. `DEF-LOG-001` — Content-Minimized Security Logging

Runtime logging preserves request, identity-independent retrieval, timing, usage,
safe-error, and security-decision metadata. Security fields include:

- `security_mode` and `security_policy_version`;
- `enabled_defense_ids`;
- signal status, action, IDs, counts, and flagged chunk IDs;
- output-validation status;
- `output_blocked`, `blocked_reason`, and `blocking_defense_id`.

The allow-list excludes:

- raw user question;
- prompt/message bodies;
- context and document content;
- answer text;
- protected canary values;
- API keys, database credentials, JWT secrets, and authorization headers;
- private exception details.

Logging is best-effort and fail-open because it records security decisions; it does not
make them. A log write failure cannot replace or authorize an answer.

### Limitations

Document/chunk IDs, ranks, model identity, usage, timing, error stage, and security
outcomes remain operationally sensitive. Runtime logging has no application-level
per-user read API, retention schedule, rotation policy, or encryption policy.

## 16. Error and Failure Policy

- Missing, invalid, expired, or stale Bearer credentials fail closed.
- Unauthorized document reads and share operations fail closed.
- An empty readable document set yields no unauthorized retrieval fallback.
- Invalid security modes fail before generation.
- Invalid protected-canary definitions fail during policy construction.
- Observe-only signal detection has no blocking or authorization authority.
- Defined output violations fail closed to a non-echoing safe response.
- Unexpected output-validator errors fail closed in layered mode.
- Provider errors retain safe public error semantics; layered validation prevents a context-serializing fallback body.
- Logging construction or write failures remain fail-open and cannot alter access or answer authorization.
- PostgreSQL authorization or metadata unavailability fails protected operations closed rather than searching a global corpus.

## 17. End-to-End Current Flow

```text
ChatRequest
-> Bearer JWT validation
-> current UserIdentity
-> readable document-ID set
-> authorized candidates
-> authorized final contexts
-> context signal summary
-> layered message dictionaries
-> configured LLM response
-> deterministic citations from final contexts
-> output and citation validation
-> authorized public response
-> content-minimized security event
```

Data trust does not increase merely because content passes upload, parsing, indexing,
authorization, retrieval, reranking, or signal checks. Authorization permits access; it
does not establish instruction authority or factual accuracy.

## 18. Threat-to-Control Mapping

| Threat | Deterministic access controls | Prompt/output controls | Residual risk |
|---|---|---|---|
| AUTH-001 | JWT validation and current-user reload | None | Credential/signing-key compromise and missing account-security features |
| ACL-001 | Owner/ACL checks and authorized retrieval | Citation membership adds defense in depth | Missed future code path or backend filter regression |
| DPI-001 | Limits available document scope | `DEF-PROMPT-001`, `DEF-OUTPUT-001`, `DEF-LOG-001` | Unseen and semantic attacks |
| IPI-001 | Only readable documents may reach context | All five `DEF-*` controls | Authorized malicious content may still influence the model |
| MD-001 | Authentication, owner assignment, ACL-limited readers | Context framing and signal | No provenance approval, quarantine, or factual verification |
| SPL-001 | Prevents unauthorized source access | Anti-disclosure hierarchy and deterministic exact checks | Paraphrased leakage |
| SIL-001 | Pre-LLM permission-aware retrieval | Unsafe fallback and exact protected-output blocking | Provider, host, artifact, semantic, or authorized-user disclosure |
| CP-001 | Restricts poisoning to accessible corpus | Framing may help | Plausible false evidence remains largely outside controls |
| CI-001 | Authorized final-context source set | Membership validation | No claim-level support validation |
| LOG-001 | Runtime logs not exposed by application routes | Content-minimized schema | Host/repository access and lifecycle policy |

## 19. Frozen Evaluation Evidence

The frozen baseline/layered comparison used the same production retrieval, reranking,
context construction, provider/model identity, generation settings, and citation
construction, while changing the security policy and output controls.

Key paired results on the known synthetic suites:

- direct final-response ASR: **30.0% (3/10) -> 10.0% (1/10)**;
- indirect end-to-end/conditional ASR: **12.5% (1/8) -> 12.5% (1/8)**;
- answerable false refusal: **8.7% (2/23) -> 4.3% (1/23)**;
- benign unanswerable abstention: **8/8 in both modes**;
- document citation exact match and mean F1 showed no regression;
- one direct output-format attack remained successful;
- plausible false-evidence context poisoning remained a residual risk.

The direct result supports lower observed attack success on the known suite. The indirect
result supports no paired improvement. The suites were known during defense development,
small, synthetic, and evaluated without confidence intervals.

### Authorization Evidence Boundary

The frozen prompt-injection evaluation predates the current authentication and ACL
implementation. Its artifacts may correctly record that authorization was not implemented
for that historical run. They must not be read as evidence that the current API still
uses a shared corpus or as a test of current cross-user isolation.

Current authentication, ownership, ACL, revocation, and permission-aware retrieval are
separate deterministic controls covered by current code/tests and architecture.

## 20. Baseline/Layered Comparability

The historical baseline prompt hash remains:

```text
5255b0fbfa95bbceb0610ddc474e4c5cc4f17b621a4d8a847332003c54126418
```

The layered prompt hash remains:

```text
2de645cdfa58d620240f0f92e9a374c35dbb305cb7c074a9ba55efd106dfc43d
```

Both historical modes used the same `answer_question` path, retrieval configuration,
Vector/BM25 candidates, RRF setting, reranker, final Top-K, context selection, provider,
model, temperature, max tokens, and citation builder. Frozen datasets and artifacts must
not be rewritten solely to remove historical identifiers because their hashes and
provenance depend on byte identity.

## 21. Current Residual Risks

- Prompt injection remains probabilistic and unseen attacks may bypass model-visible instructions.
- Permission does not imply document trust or factual correctness.
- Context poisoning by plausible false content remains outside deterministic output checks.
- Observe-only signals can miss attacks and flag legitimate text.
- Output validation covers exact deterministic contracts, not semantic correctness.
- Claim-level citation support is not validated.
- A compromised user account can upload or read content within that user's permissions.
- Every future retrieval backend must preserve allowed-document filtering.
- Remote providers may process authorized private context under policies outside this repository.
- Runtime logs and frozen evaluation artifacts require publication and access review.
- Render Free does not provide durable document/index storage or a colocated Ollama model.

## 22. Operational Review Checklist

Before a public release or security-sensitive change, verify:

- protected routes require Bearer authentication;
- current-user resolution reloads PostgreSQL state;
- document list, preview, chunks, share, search, and chat enforce owner/ACL rules;
- revocation blocks the next request;
- all vector/BM25/hybrid/rerank/context paths receive or preserve the readable-ID set;
- citations and contexts contain only authorized final chunks;
- normal traffic defaults to `layered`;
- baseline mode is limited to intentional reproducibility use;
- protected canaries, raw prompts, contexts, answers, credentials, and auth headers are absent from runtime logs;
- `.env`, runtime storage, and logs are not tracked;
- frozen machine artifacts remain byte-identical when cited by hash;
- public documentation distinguishes historical prompt-security experiments from current ACL behavior.

## 23. Review Triggers

Update this document when any of the following changes:

- JWT, user-state, password, or authentication behavior;
- ownership, ACL, sharing, revocation, or permission scope;
- document/search/chat endpoint behavior or response schemas;
- retrieval, reranking, context expansion, optional Qdrant, or citation construction;
- security policy, prompt, signal, validator, or log schema versions;
- LLM, embedding, vector-store, database, or deployment provider;
- security datasets, model identities, rubrics, or frozen result artifacts;
- artifact/log access, retention, encryption, or publication policy;
- tenant, group, write/edit, field-level, or administrative permission requirements.
