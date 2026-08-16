# Known Limitations and Failure Analysis

## Scope and Method

This document describes the failure modes, capability limits, residual risks, and
evidence gaps of the current repository version. It is a diagnosis, not a new benchmark
and not a list of generic RAG concerns. Every project-specific
observation below is tied to committed code, tests, or a frozen evaluation artifact.

The analysis follows the runtime stages:

```text
document
-> parse
-> chunk
-> retrieve candidates
-> fuse / rerank
-> build context
-> generate
-> attach citations
```

A problem observed at one stage is not reassigned to another stage without evidence.
For example, a reranker cannot recover a relevant chunk that never entered its candidate
pool, and a lexical answer-metric mismatch is not automatically a generation failure.

The four-document, 12-chunk corpus discussed here is the frozen controlled evaluation
fixture under `eval_docs/` and `storage/eval/`. It is not the mutable application corpus
created through runtime uploads. Full metric definitions, denominators, model identity,
and experiment conditions are in [Real Evaluation Results](evaluation-results.md).

## Evidence Classification

The status labels are deliberately narrow:

| Status | Meaning |
|---|---|
| `OBSERVED` | A preserved test or evaluation case exhibited the behavior. |
| `VERIFIED LIMITATION` | Code, configuration, or platform inspection establishes a current boundary. |
| `NOT IMPLEMENTED` | The repository has no implementation for the capability. |
| `NOT FORMALLY TESTED` | The capability or risk may matter, but current evidence does not establish its behavior. |

These labels prevent three common errors: treating an untested risk as a failure,
treating an intentional non-goal as a defect, and interpreting success on a small fixed
suite as a general guarantee.

## Summary Matrix

| ID | Limitation or failure | Status | Main impact |
|---|---|---|---|
| ING-01 | No OCR path | `NOT IMPLEMENTED` | Image-only PDFs provide no searchable text. |
| ING-02 | Table, image, and layout semantics are limited | `VERIFIED LIMITATION` | Visually encoded relationships may be flattened or omitted. |
| CHN-01 | Fixed character-based chunk representation | `VERIFIED LIMITATION` | Semantic and structural boundaries are not guaranteed. |
| RET-01 | q027 candidate-recall failure | `OBSERVED` | One required policy never reached the reranker. |
| RET-02 | q027 fusion/cutoff decomposition of RET-01 | `OBSERVED` | Hybrid RRF returned only half of the required documents at diagnostic K=3. |
| RET-03 | Frozen Cross-Encoder reranking adds latency without measured held-out gain | `OBSERVED` | More compute and complexity did not improve the five-query aggregate. |
| GEN-01 | Extra-context citations in five cases | `OBSERVED` | Answers cited unrelated extra documents. |
| GEN-02 | Semantic answer-quality coverage is incomplete | `NOT FORMALLY TESTED` | Keyword proxies cannot establish correctness, groundedness, or relevance. |
| SEC-01 | Direct prompt-injection residual success | `OBSERVED` | Layered defenses reduced but did not eliminate known direct attacks. |
| SEC-02 | No paired improvement for indirect injection | `OBSERVED` | Malicious retrieved content remains a residual risk. |
| SEC-03 | Security suite and leakage coverage are narrow | `VERIFIED LIMITATION` | Unseen-attack generalization is not established. |
| ACL-01 | Authorization is document-level read access | `VERIFIED LIMITATION` | Group, tenant, field, write, and edit policies are outside current scope. |
| STO-01 | Database and local artifact writes are not atomic | `VERIFIED LIMITATION` | Partial state can exist after a cross-storage failure. |
| DEP-01 | Render Free has an ephemeral filesystem | `VERIFIED LIMITATION` | Uploaded files, indexes, and logs are not durable in the cloud profile. |
| DEP-02 | No verified reachable Ollama service in the Render profile | `VERIFIED LIMITATION` | Render-profile LLM-backed RAG has not been accepted. |
| PERF-01 | No concurrency, throughput, or SLA evidence | `NOT FORMALLY TESTED` | Local latency cannot predict production capacity. |
| OBS-01 | Observability is local and historical joins are incomplete | `VERIFIED LIMITATION` | Cross-instance and historical request diagnosis is constrained. |
| EVAL-01 | Evaluation is small, saturated, and sparsely labeled | `VERIFIED LIMITATION` | Statistical significance and broad generalization are not established. |

## Document Ingestion

The application supports UTF-8/GBK TXT, Markdown, text-layer PDF, and DOCX. PDF
extraction retains page numbers. DOCX extraction includes paragraphs and basic table
cell text. Unsupported, malformed, and empty documents fail explicitly rather than
silently producing an index.

### ING-01 — No OCR path

**Status:** `NOT IMPLEMENTED`

**Evidence:** `app/services/text_loader.py` uses `pypdf` text extraction and contains no
OCR engine or image-to-text stage. The blank-PDF test verifies that a PDF with no
extractable text becomes `EmptyDocumentError`.

**Mechanism:**

```text
image-only PDF
-> pypdf finds no embedded text
-> no OCR fallback exists
-> no extracted sections
-> document rejected as empty
```

**Impact:** A scanned policy cannot contribute searchable chunks merely because it has a
`.pdf` extension.

**Current mitigation:** The upload pipeline rejects empty extraction explicitly, which
prevents a misleading successful ingestion with zero useful content.

**Residual limitation:** OCR quality, language support, confidence, page-image handling,
and OCR-specific security risks have not been evaluated.

**Possible future work:** Add an opt-in OCR stage with synthetic scanned fixtures,
confidence thresholds, resource limits, and clearly separated text-layer/OCR provenance.

### ING-02 — Table, image, and layout semantics are limited

**Status:** `VERIFIED LIMITATION`

**Evidence:** DOCX table cells are joined as plain text with ` | ` separators. PDF
extraction reads page text, not table structure or coordinates. Neither loader extracts
embedded images or preserves columns, heading hierarchy, merged cells, or relationships
between figures and captions.

**Impact:** The words in a simple table may remain searchable while the row/column
relationship that gives them meaning may be weakened. Information present only in an
image is omitted.

**Current mitigation:** PDF chunks retain page provenance, DOCX table text is not wholly
dropped, and original uploaded bytes remain available in local storage while the
instance lives.

**Residual limitation:** No formal corpus of complex tables, multi-column PDFs, diagrams,
forms, headers/footers, or nested DOCX structures has been evaluated. Their accuracy is
therefore `NOT FORMALLY TESTED`, not an observed failure rate.

**Possible future work:** Evaluate representative structured documents before selecting
a layout-aware parser. Preserve structural metadata rather than flattening it directly
into anonymous text.

## Chunking and Representation

### CHN-01 — Fixed character-based chunks do not model document structure

**Status:** `VERIFIED LIMITATION`

**Evidence:** `split_text()` defaults to 500 characters with 100 characters of overlap.
It searches backward for paragraph, newline, English sentence, or whitespace boundaries
in the latter half of the target window. If none is found, it cuts at the character
limit. PDF pages are processed as separate sections.

**Impact:** Character count is not token count. Headings, lists, table rows, clauses, and
multilingual sentence boundaries do not necessarily align with the selected boundary.
Overlap also duplicates text and can increase context noise.

**Current mitigation:** Boundary preference avoids many arbitrary mid-sentence splits;
overlap preserves some cross-boundary context; chunks carry document, position, page,
and creation metadata.

**Residual limitation:** The formal artifacts do not causally attribute any current
quality failure to the splitter. In particular, q027 is a saved candidate-recall and
cutoff failure, not proven chunking damage. Multi-page reasoning, complex tables,
multilingual punctuation, very long unbroken strings, and alternative chunk sizes have
not been formally compared.

**Possible future work:** First create structure-sensitive ground truth, then compare
token-aware or section-aware chunking without changing the existing frozen baseline.

## Retrieval and Ranking

### RET-01 — Candidate recall bounded q027

**Status:** `OBSERVED`

**Evidence:** q027 asks for both working-abroad approval and customer-data
classification rules. Its strict relevant chunks are one HR chunk and one Security
chunk. In the saved broader five-result Hybrid RRF ranking, the Security chunk was
ranked first and the HR chunk fourth. The frozen retrieval and generation path passed
only the top three Hybrid RRF results to the reranker, so the Security chunk entered that top-3 reranker
candidate set and the required HR chunk did not.

**Mechanism:**

```text
multi-policy query
-> Vector + BM25 candidates
-> broader RRF ranking places required HR chunk fourth
-> select top three as the reranker candidate set
-> required HR chunk excluded from reranker input
-> reranker never receives HR chunk
-> final context contains only Security evidence
-> answer incorrectly abstains
```

**Impact:** Downstream reranking and generation cannot use evidence they never receive.
This is the one confirmed strict-chunk retrieval-primary failure in the frozen
failure-analysis artifact.

**Current mitigation:** Hybrid retrieval combines lexical and semantic candidates, and
the pipeline preserves pre-reranker candidates for diagnosis.

**Residual limitation:** The controlled dataset contains only one multi-relevant
held-out query. This mechanism is confirmed for q027 but its frequency elsewhere is
unknown.

**Possible future work:** Add unseen multi-document questions and measure candidate
recall separately from final ranking before tuning candidate depth.

### RET-02 — Fusion and cutoff did not preserve all relevant evidence

**Status:** `OBSERVED`

**Evidence:** At diagnostic K=3, BM25 ranked q027's two relevant documents within the
cutoff. Vector and Hybrid RRF recalled only one. The broader Hybrid RRF ranking placed
the required HR chunk fourth, immediately outside the top-3 reranker candidate set.

RET-02 is a diagnostic decomposition of the same q027 root failure described by RET-01,
not a second independent observed failure. RET-01 describes the downstream candidate-
recall ceiling; RET-02 identifies how the fused rank and top-3 cutoff created that
ceiling.

**Impact:** RRF successfully combined rankings but did not improve q027's completeness.
This is not evidence that fusion universally regresses: Hybrid and Vector were equal on
the other measured document-level outcomes.

**Current mitigation:** Per-source candidate depth is five and source ranks/scores are
retained. The deployed evaluation configuration returns `final_top_k=2`; the K=3 value
is a pre-final post-rerank diagnostic, not the operational response cutoff.

**Residual limitation:** The retrieval study did not isolate query mechanisms such as
abbreviations, identifiers, exact terms, or semantic paraphrases. The largely single-relevant,
metric-saturated dataset cannot establish when lexical, vector, or fused retrieval is
generally superior.

**Possible future work:** Build a frozen mechanism-labeled dataset before changing RRF
weights, candidate depth, or operational top-k.

### RET-03 — Reranker quality/latency trade-off

**Status:** `OBSERVED`

**Evidence:** On five held-out queries, Vector, Hybrid RRF, and Hybrid + Reranker all had
Hit@2 1.0, Recall@2 0.9, and MRR@2 1.0. The Cross-Encoder changed some ordering but not
the aggregate. Warm local P50 rose from 16.138 ms for Hybrid to 64.352 ms with reranking,
an increase of 48.213 ms. The isolated reranker-stage P50 was 49.000 ms.

**Impact:** The current evidence does not justify making this Cross-Encoder path the
default: it adds model loading, CPU inference, memory, failure modes, and latency
without a measured held-out quality gain.

**Current mitigation:** Reranking is explicitly selectable through the public chat
`rerank` mode. That public mode uses the repository's explainable heuristic reranker and
is distinct from the frozen `hybrid_rerank` evaluation path that uses the
Cross-Encoder measured here. The current held-out benchmark does not justify preferring
that Cross-Encoder path solely on measured quality given its added latency. One
discarded warm-up and 25 measured sequential samples per method separate warm
measurements from startup.

**Residual limitation:** This is not proof that reranking never helps. The held-out set
is small and saturated. The single cold Vector observation also paid embedding-model
initialization and is not a fair cold method comparison.

**Possible future work:** Reassess on a larger, harder held-out set with candidate-recall
diagnostics, memory measurements, and a defined latency budget.

## Generation and Citations

### GEN-01 — Pipeline citations can over-credit context

**Status:** `OBSERVED`

**Evidence:** The frozen failure analysis identified q006, q015, q017, q019, and q020 as
citation-primary failures. Their answers were assessed as acceptable, but context expansion retained an
unrelated second document and the pipeline mechanically returned that document in the
citation set. q027 also had a contributing missing-source citation failure because the
required HR evidence was absent upstream.

**Mechanism:**

```text
relevant final result + lower-relevance result
-> context expansion
-> answer uses the relevant evidence
-> citation builder cites all finalized contexts
-> unrelated extra document appears in response citations
```

**Impact:** Document citation exact-match and precision decrease, and a reader may infer
that every cited document supports the answer.

**Current mitigation:** Citations are deterministic and traceable to chunk/document
metadata; the model does not invent opaque source strings.

**Residual limitation:** The current logger and artifacts do not provide claim-to-source
attribution. Therefore the evidence supports citation over-inclusion, not a claim that
the LLM fabricated citations.

**Possible future work:** Define claim-level citation ground truth and select citations
from answer-support alignment rather than all supplied contexts.

### GEN-02 — Semantic answer-quality coverage is incomplete

**Status:** `NOT FORMALLY TESTED`

**Evidence:** The generation evaluation uses deterministic expected-keyword proxies. q005 was
flagged because expected `3-month` differed lexically from actual `3 months`; manual
analysis found the answer semantically acceptable. Groundedness and answer relevance
were not formally automated. The frozen failure analysis found zero generation-primary failures among its
selected cases, but that is not a general hallucination result.

**Impact:** Aggregate keyword scores can under- or over-estimate answer correctness, and
the repository cannot claim comprehensive semantic faithfulness.

**Current mitigation:** Deterministic metrics remain reproducible, raw answers and
contexts are preserved, and uncertain diagnoses can be marked `needs_review` instead of
forced into a failure category.

**Residual limitation:** The eight unanswerable evaluation cases all produced correct
abstention with no misleading citation, and four answerable controls produced no false
abstention. This establishes behavior only on that fixed suite; it does not imply zero
hallucination on unseen questions.

**Possible future work:** Introduce a versioned semantic rubric with human review or a
separately validated evaluator while retaining deterministic proxies as diagnostics.

## Security Residual Risks

The security comparison used the same formal Ollama `qwen3:8b` model identity and fixed
retrieval configuration before and after defenses. The full threat boundaries and
controls are documented in [Threat Model](security/threat-model.md) and
[Layered Defenses](security/layered-defenses.md).

### SEC-01 — Direct prompt injection was reduced, not eliminated

**Status:** `OBSERVED`

**Evidence:** On the ten paired common successful executions, final-response direct
attack success fell from 3/10 to 1/10. One known direct attack therefore still succeeded,
and one case deteriorated from resisted to partial success under the broader regression
view.

**Impact:** Prompt hardening, input signals, and output validation reduce risk but do not
create a proof of instruction hierarchy or safe behavior for every attack.

**Current mitigation:** Layered prompt rules, context delimiting, detection signals,
output validation, safe logging, and frozen regression tests.

**Residual limitation:** The suites were known during defense development, are small,
and have no statistical significance or unseen-attack generalization claim.

**Possible future work:** Preserve the current suite as regression evidence and evaluate
separately curated unseen attacks without redefining the frozen security baseline.

### SEC-02 — Indirect injection showed no paired improvement

**Status:** `OBSERVED`

**Evidence:** For eight paired common indirect executions, end-to-end attack success was
1/8 for both baseline and layered modes. The broader non-paired final-response rates are
not the primary comparison because execution populations differ.

**Impact:** A malicious document that enters authorized retrieval can still influence
generation. ACL prevents unauthorized documents from entering retrieval; it does not
make authorized malicious content trustworthy.

**Current mitigation:** The pipeline labels retrieved content as untrusted, records
security signals, and validates output. Authorization filters documents before context
construction.

**Residual limitation:** No observed paired improvement means indirect injection remains
an explicit residual risk. It does not mean every indirect attack succeeds.

**Possible future work:** Expand the unseen malicious-document suite and evaluate
content isolation, provenance, and output-policy enforcement independently.

### SEC-03 — Leakage and attack-suite evidence are narrow

**Status:** `VERIFIED LIMITATION`

**Evidence:** Known direct prompt-extraction success moved from 1/2 to 0/2, although one
layered result remained partial. The indirect document-canary case did not leak, but that
canary is untrusted document content, not a protected system-prompt secret. Neither
frozen prompt contains a protected prompt canary, so protected-canary leakage is not
applicable rather than proven absent.

**Impact:** The project cannot claim that system prompt leakage, context poisoning, or
prompt injection is solved.

**Current mitigation:** Explicit claim boundaries, threat model, fixed model identity,
frozen attack artifacts, and separate direct/indirect reporting.

**Residual limitation:** The prompt-injection evaluation predates the authorization
implementation and therefore does not test cross-user isolation. Permission regression
tests provide ACL evidence but cannot replace prompt-injection testing.

## Authentication and Authorization Scope

### ACL-01 — Correct document-level read ACL, deliberately narrow scope

**Status:** `VERIFIED LIMITATION`

The following are verified capabilities, not failures:

- Bearer authentication reloads the current PostgreSQL user on each request.
- JWT claims carry identity and timing, not document permissions.
- Owners have implicit read access; explicit active grants allow shared read.
- Default is deny, and revocation affects the next request.
- Document list, preview, chunks, search, and RAG filter before returning content.
- Alice/Bob/Carol matrix tests verify owner/shared/unrelated behavior and 401/403/404
  semantics.

The current permission model is document-level read access. It does not implement group,
tenant, department, role, field-level, chunk-level, write, edit, or delete policies.
Those absent capabilities are scope limits, not evidence that the existing read ACL is
broken.

Future permission work should start from an explicit product requirement and preserve
the invariant:

```text
authenticate
-> resolve current database permissions
-> restrict allowed documents
-> retrieve
-> construct context
-> call LLM
```

## Storage and Deployment

### STO-01 — PostgreSQL and local artifact writes are not atomic

**Status:** `VERIFIED LIMITATION`

PostgreSQL persists users, document metadata, ownership, and ACL rows. Original uploads,
extracted text, chunk content, FAISS/NumPy/JSON indexes, and structured RAG logs live on
the filesystem. These stores do not participate in one distributed transaction.

**Impact:** A process or storage failure between file/index writes and database commit can
leave partial state requiring reconciliation.

**Current mitigation:** Explicit write errors, metadata conflict checks, fail-closed
authorization, and upload retry replacement for a document's chunks.

**Residual limitation:** The repository has no outbox, two-phase commit, background
reconciler, or durable object-store transaction protocol.

**Possible future work:** Define the durable storage architecture first, then add an
idempotent ingestion state machine and reconciliation process.

### DEP-01 — The free cloud filesystem is ephemeral

**Status:** `VERIFIED LIMITATION`

The selected profile is Render Free Docker plus Neon Free PostgreSQL 18. Public HTTPS,
container startup, migrations, health, and PostgreSQL-backed registration/login
persistence were verified. Render restart, redeploy, or idle lifecycle events do not
guarantee persistence for local uploads, extracted text, indexes, or JSONL logs.

**Impact:** A document row can survive in Neon while its corresponding local RAG
artifacts disappear. The public deployment is therefore not a durable document service.

**Current mitigation:** The boundary is explicit in the
[Deployment Guide](deployment.md); no real document is represented as durably stored in
this profile. The optional Qdrant adapter is not the active deployed backend.

**Residual limitation:** Render Free can idle-spin down, and Neon can scale to zero. The
first request may be slower. A controlled restart preserved PostgreSQL login state, but
an automatic idle cold start was not timed end to end.

**Possible future work:** Use durable object storage and a managed/persistent vector
backend before claiming cloud RAG persistence.

### DEP-02 — The Render profile has no verified reachable Ollama inference service

**Status:** `VERIFIED LIMITATION`

LLM-backed RAG is implemented in the repository through the RAG service and shared
OpenAI-compatible LLM client. The deployment limitation is narrower: Render Free does
not run a colocated Ollama instance and has no verified reachable Ollama endpoint. Local
development uses `gemma3:4b`; formal generation evaluation uses the frozen `qwen3:8b`
identity. Neither model was accepted in the deployed free profile.

**Impact:** A healthy public `/health` endpoint proves web-process availability, not a
complete cloud retrieval-to-generation path.

**Current mitigation:** LLM-backed routes retain explicit provider-unavailable behavior,
and deployment documentation makes no cloud-generation claim.

**Possible future work:** Select an explicitly authorized inference deployment that can
meet model, privacy, cost, and network requirements, then run a separate acceptance.

## Performance and Observability

### PERF-01 — Local latency is not a production SLA

**Status:** `NOT FORMALLY TESTED`

The latency benchmark measured one local Windows CPU process, sequential requests, a
fixed order, five held-out queries, and 25 warm samples per method. It excluded LLM
generation, concurrency, throughput, saturation, queueing, multi-worker behavior, cloud
networking, and service availability.

Cold diagnostics included a 10,525.938 ms first Vector call and a 207.805 ms first
reranker stage. These single observations combine lazy initialization and first use;
they are diagnostic startup costs, not production distributions or a fair cold
cross-method comparison.

**Impact:** The project has no evidence-backed requests-per-second capacity, concurrent
user limit, latency objective, availability objective, or monetary cost model.

**Current mitigation:** Warm/cold evidence is separated, failures and sample counts are
preserved, and no production SLA claim is made.

**Possible future work:** Define a workload and SLO before running concurrent load,
resource, tail-latency, and cost measurements in the intended deployment environment.

### OBS-01 — Logs are useful locally but not centralized

**Status:** `VERIFIED LIMITATION`

The runtime writes one content-minimized JSONL event per RAG request with a UUID request
ID, model/provider identity, retrieved/context/cited IDs, ranking scores, stage timings,
token usage, safe errors, and security signals. It deliberately excludes raw query,
prompt, answer, context text, document content, and secrets.

Limitations are:

- `embedding_ms` remains null because embedding is not independently timed;
- files are local and ephemeral in the Render profile;
- no centralized aggregation, dashboards, alerting, retention policy, or cross-instance
  correlation is implemented;
- historical generation and abstention artifacts predate request IDs, while the
  corresponding runtime logs lack evaluation run/query IDs, so request-level historical
  joins were 0/12;
- no claim-level model-attribution trace exists.

**Impact:** A current local request can be diagnosed more effectively than a historical
or multi-instance production incident.

**Possible future work:** Export the same minimized schema to durable centralized
telemetry and add evaluation correlation IDs without logging sensitive content.

## Evaluation Evidence

### EVAL-01 — Small, saturated, sparsely labeled evidence

**Status:** `VERIFIED LIMITATION`

The controlled corpus contains four documents and 12 chunks. The parent dataset has 27
queries. The retrieval-configuration study used 18 tuning queries and five held-out answerable
queries; the held-out queries were reserved from tuning but still came from the same parent dataset.

The primary relevance judgment is document-level. Only three parent queries have
strict chunk labels, and only q027 is strict-chunk labeled in the five-query held-out
split. Twenty-two of 23 answerable parent queries have only one relevant document.
Document-level Hit and MRR are saturated across methods, and stable graded relevance for
nDCG is unavailable.

For the generation evaluation, the 23 successful answerable cases are the denominator
for ordinary retrieval, expected-keyword, and document-citation metrics. The strict chunk metrics use only the
three labeled cases. The four no-context parent queries are excluded from ordinary
metrics and evaluated with additional cases in the unanswerable evaluation.

**Impact:** The artifacts are reproducible and useful for regression and case diagnosis,
but they do not establish statistical significance, performance on a mutable runtime
corpus, or generalization to unseen organizations, languages, document formats,
questions, and attacks.

**Current mitigation:** Tuning/held-out identity, source hashes, model identity,
denominators, K semantics, raw cases, and claim boundaries are preserved. K=2 is the
headline operational cutoff. K=3 is explicitly a pre-final post-rerank diagnostic; it is
not presented as a deployed three-result response.

**Possible future work:** Add a separately frozen, unseen, larger evaluation suite with
more multi-relevant cases, stronger chunk/claim labels, mechanism categories, and a
predeclared statistical analysis plan.

## Explicit Non-goals

The absence of the following is not classified as a project defect because the current
project scope does not claim these capabilities for the reference implementation:

- Kubernetes, microservices, GraphRAG, or a knowledge graph;
- a general-purpose document-management product;
- production multi-tenancy or enterprise identity federation;
- a paid, autoscaled, high-availability cloud platform;
- proof that one generation model or retrieval technique is globally optimal;
- a claim that prompt injection can be eliminated by prompt wording alone.

## Prioritized Future Improvements

These are directions, not commitments or completed fixes:

1. Improve evidence first: build a larger unseen suite with multi-relevant, strict-chunk,
   semantic-answer, claim-citation, and unseen security labels.
2. Address candidate recall before adding more reranker complexity.
3. Evaluate claim-aware citation selection against the existing over-inclusion cases.
4. Add OCR/layout parsing only with representative fixtures and resource/security limits.
5. Establish durable object/vector storage and idempotent ingestion before claiming a
   persistent cloud RAG service.
6. Define a target workload and SLO before concurrency, capacity, and cost testing.
7. Centralize the existing minimized event schema without adding sensitive content.

## What Is Proven

- Supported text-bearing TXT/Markdown/PDF/DOCX fixtures follow the tested extraction and
  chunk pipeline; malformed and empty cases fail explicitly.
- q027's required HR strict chunk ranked fourth in the broader Hybrid RRF ranking and
  was therefore absent from the top-3 reranker candidate set.
- Five evaluated cases contained acceptable answers plus unrelated extra document
  citations.
- On the frozen five-query held-out split, the reranker added measured warm latency
  without changing the reported aggregate retrieval metrics.
- On paired known attacks, layered defenses reduced direct final-response success but did
  not change indirect final-response success.
- Owner and explicit read ACLs filter document, retrieval, and RAG access before content
  reaches the LLM, with immediate revocation on the next request.
- Neon persisted relational identity/permission state in the deployment acceptance;
  Render's local RAG artifacts are not durable by contract.

## What Is Not Proven

- OCR, complex table, image, layout, or multi-page reasoning quality;
- that q027's failure frequency represents other corpora;
- that Hybrid or reranking is universally better or worse than another method;
- semantic correctness, groundedness, answer relevance, or zero hallucination in general;
- elimination of direct, indirect, leakage, or context-poisoning attacks;
- production concurrency, throughput, latency, memory, cost, availability, or SLA;
- durable cloud document ingestion or cloud Ollama-backed generation;
- statistical significance or generalization beyond the frozen controlled fixtures.

## Evidence / Source Map

- [Architecture](architecture.md)
- [Real Evaluation Results](evaluation-results.md)
- [Deployment Guide](deployment.md)
- [Threat Model](security/threat-model.md)
- [Layered Defenses](security/layered-defenses.md)
- [Retrieval failure-analysis artifact](../evals/results/W6-T5-failure-analysis.json)
- [Latency trade-off artifact](../evals/results/W7-T4-latency-trade-off.json)
- [Generation failure-analysis artifact](../evals/results/failure_analysis_runs/w8-t4-20260806T084416807510Z-qwen3-8b.json)
- [Security comparison artifact](../evals/results/security/security_evaluation_runs/w9-t5-20260808T182607179698Z-qwen3-8b/comparison.json)
