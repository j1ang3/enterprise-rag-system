# Real Evaluation Results

This page consolidates the project's frozen retrieval, generation, abstention, latency,
and security evidence. No additional tuning or formal benchmark rerun was performed.
The repository contains public-redacted derivatives of the original machine-readable
artifacts: machine-specific paths and private process-document references were removed,
while datasets, model outputs, metrics, case labels, configurations, and model identities
were preserved. The transformation is recorded in
[the public artifact manifest](../evals/public_manifest.json).

The measurements describe a small controlled corpus and fixed local experiments. They do
not establish statistical significance, unseen-data generalization, production
throughput, or a service-level objective.

## Evidence provenance

| Evidence | Role in this page | SHA-256 |
|---|---|---|
| [Retrieval baseline](../evals/results/W6-T4-retrieval-evaluation.json) | Supplementary full answerable-set comparison | `3c26abc2165f71ba3a3438b23cd4fdcd00720b223e702e370e95591cd5c8df77` |
| [Selected retrieval configuration](../evals/results/W7-T2-retrieval-configuration.json) | Frozen configuration-selection evidence | `13620ba452961676fde43bc74d902db3f94ea53821c973a3ab2655de48cbdbdc` |
| [Held-out retrieval quality](../evals/results/W7-T3-retrieval-evaluation.json) | Primary held-out quality comparison | `6c513fad336df0482f401c0f33ca6cfeb48cd749e71fe74f01d7826aea195391` |
| [Retrieval latency](../evals/results/W7-T4-latency-trade-off.json) | Primary local latency comparison | `3114a04bcc043aeea65ab291a65ae34d3125b82e4308cb4e4cb6c429999226f9` |
| [Generation and citation evaluation](../evals/results/evaluation_runs/w8-t1-20260805T135615169893Z-qwen3-8b.json) | Formal answer and citation evidence | `50cb97bfad1db4104b5ac563e464e61383aad342852b70738f831ff186e47d29` |
| [Abstention evaluation](../evals/results/unanswerable_runs/w8-t2-20260805T163627033704Z-qwen3-8b.json) | Formal abstention evidence | `a60785baf3e2f5ea85494707bf7963e77feab38b8fb585a1bd967bd0e047b13f` |
| [Security comparison](../evals/results/security/security_evaluation_runs/w9-t5-20260808T182607179698Z-qwen3-8b/comparison.json) | Formal baseline/layered security comparison | `c5d30c0f39f0c13622026ae53b46f16bb1ffcc56e05e43683d8708e5a12711e5` |

An earlier generation baseline was also audited. It is not used as the final retrieval
comparison because its retrieval score is a source-level proxy and the later formal
generation evaluation provides stronger evidence. The full answerable-set retrieval
comparison remains useful as supplementary evidence, but it is not mixed into the
held-out table.

## Dataset and ground truth

The parent evaluation dataset contains 27 queries over four documents and 12 chunks:

- 23 answerable queries;
- four unanswerable queries;
- 18 answerable tuning queries used for retrieval configuration selection;
- five reserved answerable queries used for held-out quality and latency evaluation;
- four unanswerable queries excluded from relevance tuning.

Those four documents and 12 chunks are the frozen, controlled evaluation fixture under
`eval_docs/` and `storage/eval/`. They are not the mutable runtime application corpus
created by user uploads. Formal evaluation validates the fixture's document, chunk,
vector, FAISS, and metadata hashes and refuses to bootstrap replacement documents. The
normal application instead reads its configured runtime storage and applies current
ownership/ACL filtering; its changing contents are not represented by these benchmark
denominators.

The dataset SHA-256 is
`9ebbe2fca70b195e73d42ce3a795a01bc8bfd01680d148302bcf7442d14925e4`.
Primary relevance labels identify relevant documents. Only three cases in the parent
dataset have strict relevant-chunk labels, and only `q027` has such labels in the
held-out split. The headline Recall and MRR results are therefore document-level
retrieval metrics, not a broad chunk-level benchmark.

The retrieval configuration was selected using only the 18-query tuning split. The
five-query held-out split was not inspected during selection. Both splits still come from the same
small parent dataset, which limits generalization.

## Frozen evaluation configuration

The selected retrieval configuration was:

```text
configuration ID:          ps05-r060-n03-k02
per-source candidate depth: 5
RRF k:                      60
rerank candidates:          3
final top_k:                2
```

This is the frozen configuration for the final retrieval and generation experiments.
It did not replace every public API runtime default: the API continues to expose
`keyword`, `vector`, `hybrid`, and `rerank` retrieval modes with their documented
runtime configuration.

## Retrieval quality

The primary comparison uses the same five held-out answerable queries, corpus, labels,
candidate budget, and metric implementation for all three systems. The operational
cutoff is the frozen `final_top_k=2`.

| System | Hit Rate@2 | Recall@2 | MRR@2 | Quality queries |
|---|---:|---:|---:|---:|
| Vector | 1.0000 | 0.9000 | 1.0000 | 5 |
| Hybrid RRF | 1.0000 | 0.9000 | 1.0000 | 5 |
| Hybrid RRF + Cross-Encoder | 1.0000 | 0.9000 | 1.0000 | 5 |

Metrics at K=1 were also unchanged. The offline evaluator retained a depth-three
ranking for every method; for Hybrid + Reranker, this is the three-item post-rerank
ordering before the operational `final_top_k=2` truncation. Its diagnostic K=3 metrics
were likewise unchanged, but they do not describe the configured response cutoff. K=2
therefore remains the headline operational cutoff.

Recall@5 is not reported for this three-system comparison because the frozen reranker
candidate budget is three. Increasing it to five would change the evaluated system.
nDCG is also not reported because the current dataset does not provide stable
graded-relevance labels.

Reranking changed the order of results for `q005` and `q015`, but those changes did not
alter Hit Rate, Recall, or MRR. For `q027`, the second relevant document did not enter
the three-item candidate pool, so the reranker could not recover it. This is a candidate
recall failure rather than evidence that the Cross-Encoder chose incorrectly among all
possible documents.

As supplementary evidence, the full answerable-set comparison evaluated all 23
answerable queries before the held-out protocol. At K=3, Vector, BM25, and Hybrid RRF
recorded Recall values of `0.9783`, `1.0000`, and `0.9783`; all three recorded MRR of
`1.0000`. These supplementary numbers are not placed in the primary table because the
split and experiment protocol differ.

## Local retrieval latency

The latency benchmark measured the same frozen methods and candidates as the held-out
quality comparison. The benchmark used one
process and sequential requests on Windows 11 with Python 3.14.3, CPU inference, and
CUDA disabled. It ran one excluded full warm-up followed by five measured passes over
the five held-out queries, producing 25 successful warm samples per method.

| System | Mean | P50 | P95 | Warm samples | Failures |
|---|---:|---:|---:|---:|---:|
| Vector | 15.243 ms | 15.190 ms | 18.299 ms | 25 | 0 |
| Hybrid RRF | 15.701 ms | 16.138 ms | 18.561 ms | 25 | 0 |
| Hybrid RRF + Cross-Encoder | 65.680 ms | 64.352 ms | 69.119 ms | 25 | 0 |

Hybrid added `0.948 ms` (`6.24%`) to Vector P50 in this run. Adding the reranker to
Hybrid added `48.213 ms` (`298.75%`) to P50, while the primary quality metrics remained
unchanged. The reranker stage itself had a P50 of `49.000 ms`.

Cold diagnostics were deliberately excluded from warm aggregates. The first Vector
call, including embedding-model initialization, took `10,525.938 ms`; the first
reranker stage, including lazy loading and first inference, took `207.805 ms`. These
single cold observations are diagnostic startup costs, not a fair method comparison.

The benchmark excludes LLM generation, file I/O, metric computation, concurrency,
throughput, and saturation. It supports a local relative latency trade-off only.

## Generation and citations

Formal generation used the same RAG pipeline and the frozen Hybrid + Reranker
configuration. The model was `qwen3:8b`, resolved to the recorded Ollama digest
`500a1f...8b41`, with temperature `0.2`, maximum 512 output tokens, and no reasoning
mode. This consolidated report made no new LLM calls.

All 27 formal generation evaluation cases completed successfully: 23 answerable cases
used the model and four no-context cases used the deterministic insufficient-context
path.

The ordinary generation-quality denominator is the 23 successful answerable cases. The
four successful unanswerable cases are preserved in the artifact but excluded from
these aggregates; their behavior is evaluated separately in the abstention evaluation.
There were no execution failures in this run. Metric scope is:

| Metric family | Denominator | Scope |
|---|---:|---|
| Document-level retrieval at K=1/2 | 23 | Macro-average over successful answerable cases; expected document filenames are ground truth |
| Strict chunk retrieval at K=1/2 | 3 | Only answerable cases with explicit human-authored expected chunk IDs |
| Expected-keyword proxy | 23 | Case-insensitive substring checks against human-authored required keywords; macro-averaged over successful answerable cases |
| Document citation | 23 | Per-case set comparison of cited filenames with expected document filenames; exact-match rate and macro mean precision/recall/F1 |
| Strict chunk citation recall | 3 | Only cases with explicit human-authored expected citation chunk IDs |

| Metric | Result |
|---|---:|
| Retrieval Hit Rate@2 | 1.0000 |
| Retrieval Recall@2 | 0.9783 |
| Retrieval MRR@2 | 1.0000 |
| Expected-keyword match rate | 0.8696 |
| Expected-keyword mean recall | 0.9130 |
| Document-citation exact match | 0.7391 |
| Citation precision | 0.8913 |
| Citation recall | 0.9783 |
| Citation F1 | 0.9130 |

The expected-keyword all-required-keywords match rate is `20/23 = 0.8696`; its `0.9130`
mean recall is the macro-average of each case's matched-required-keyword fraction. The
document-citation exact-match rate is `17/23 = 0.7391`; citation precision, recall, and
F1 are macro means across the same 23 cases, rather than one pooled citation set.

Expected-keyword scores are deterministic lexical proxies, not semantic
answer-correctness judgments. Groundedness and answer relevance were not automatically
scored. Strict chunk-citation recall was `0.8333`, but its denominator contains only the
three cases with chunk-level labels.

The generation artifact records that its historical repository checkout was dirty.
Its dataset, corpus, index, configuration, provider, model, and digest identities are
still recorded, but the dirty-worktree provenance is a reproducibility limitation.

## Unanswerable behavior

The separate abstention suite contains eight unanswerable cases and four answerable
controls:

| Behavior | Result |
|---|---:|
| Correct unanswerable abstention | 8/8 (100%) |
| Unsupported or misleading citations on abstentions | 0/8 |
| False abstention on answerable controls | 0/4 |

This is evidence for the fixed suite only. It does not establish a general hallucination
rate for arbitrary questions.

## Security evaluation

The security evaluation used the same `qwen3:8b` digest and fixed generation
parameters for baseline and layered modes. Reported attack-success rates use paired
cases with valid executions in both modes, avoiding a comparison between different denominators.

| Evaluation | Baseline | Layered | Paired cases | Observed change |
|---|---:|---:|---:|---:|
| Direct prompt-injection ASR | 30.0% | 10.0% | 10 | -20.0 percentage points |
| Indirect prompt-injection ASR | 12.5% | 12.5% | 8 | 0.0 percentage points |
| Benign false-refusal rate | 8.70% | 4.35% | 23 per mode | -4.35 percentage points |
| Unanswerable strict abstention | 100% | 100% | 8 per mode | 0.0 percentage points |

The direct result shows lower observed attack success on the known suite. The indirect
result shows no observed improvement. The suite was known during defense development,
is small, and does not support statistical-significance or unseen-attack claims.
Layered defenses reduce selected risks; they do not solve prompt injection.

The frozen security evaluation predates the authentication and ACL implementation. Its
artifact correctly records `authorization_not_implemented=true` for that historical
evaluation. It must not be
read as a test of current cross-user isolation. The current system separately enforces
authentication, ownership, ACL filtering, and authorization-aware retrieval before
context reaches the LLM.

## Interpretation

The strongest conclusion supported by the frozen evidence is not that the most complex
retriever won. On the five-query held-out set, Vector, Hybrid RRF, and Hybrid +
Reranker had identical aggregate quality. Hybrid added little P50 latency, while the
Cross-Encoder added about 48 ms without an observed metric gain. The reranker did alter
some orderings, but candidate recall constrained what it could improve.

The generation pipeline produced strong deterministic retrieval and citation-recall
signals on the fixed generation suite, while citation exact match remained imperfect
and semantic groundedness was not automatically judged. Security defenses reduced direct
attack success in the known paired sample but did not improve indirect attack success.
These mixed results are more useful than a single composite score because they expose
where the current pipeline helps, where it costs latency, and where evidence remains
insufficient.

## Reproducibility and limitations

- Formal evaluation model: `qwen3:8b`; development model policy: `gemma3:4b`.
- This consolidated report reuses existing evidence and does not mix model identities
  or rerun formal evaluation.
- Artifact hashes above identify the public-redacted files; `evals/public_manifest.json` records both original and public SHA-256 values.
- Held-out retrieval quality and latency are comparable to each other; the full
  answerable-set retrieval comparison and earlier generation baseline are supplementary
  historical evidence and are not spliced into the primary table.
- The corpus, held-out set, attack suites, and strict chunk-label denominator are small.
- The held-out split was reserved during tuning but belongs to the same parent dataset.
- Latency is local and sequential; no concurrency, load, memory, monetary-cost, or SLA
  measurement was performed.
- No production cloud LLM acceptance is represented by these local qwen results.
- No result proves prompt-injection immunity, universal answer correctness, or
  production readiness.
