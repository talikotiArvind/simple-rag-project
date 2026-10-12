# Requirements — Unified Retrieval Layer (RAG + OKF)

Source: project plan dated 2026-10-12 (revised after an 8/10 technical review). Seeded directly from the
plan, which is the plan of record. Architect stage is satisfied by the plan; re-run `architect` only
for a new feature outside it.

## Problem Statement
Claude, OpenAI and Gemini agents each need grounded, citable answers over the same knowledge. Build one
retrieval layer, in about 8 weeks across phases 0–5, that combines a curated OKF knowledge bundle (Tier 1)
with hybrid RAG over raw documents (Tier 2). Agents never touch embeddings or indexes; they call a small set
of tools, and everything behind those tools is shared, so for the same query and scope every agent receives
the same evidence.

## Functional Requirements
### Ingest (Tier 2)
- FR-01: Parse PDF, Markdown and HTML sources from S3.
- FR-02: Heading-aware chunking with overlap.
- FR-03: Stable document IDs and content hashes; ingest is idempotent and retry-safe.
- FR-04: Document deletion and stale-chunk cleanup when a source changes or is removed.
- FR-05: Versioned index with rollback; every response reports `index_version`.
- FR-06: Embed with OpenAI text-embedding-3-large; store model name + version per vector.
- FR-07: Contextual retrieval — Claude Haiku writes a 1–2 sentence context per chunk with the full document
  prompt-cached; only changed documents are contextualized. Kept only if it beats the baseline (Phase 2).

### Knowledge layer (Tier 1, OKF)
- FR-08: Load and validate an OKF bundle stored in Git (`knowledge/`); `okf validate` + provenance checks in CI.
- FR-09: LLM drafts concept files from sources, each claim linked to its sources; a human flips `verified: true`.
- FR-10: Index concepts (BM25 + vectors) and build a link graph with typed edges.
- FR-11: Bounded graph expansion: 1 hop, allowed edge types only, authorization checked per node,
  at most `max_expansions` added, labelled `via="graph"`, never displacing the top_k direct hits.

### Retrieval
- FR-12: Hybrid search (BM25 + k-NN in OpenSearch) fused with reciprocal rank fusion, then reranked.
- FR-13: Evidence-aware rerank: relevance, verification, freshness and source authority weighed together.
  Verification is a ranking feature, never an override.
- FR-14: Queries are embedded server-side with the index's embedding model, whichever agent asks.
- FR-15: Every result carries `result_id` (`chunk_id` or `concept_id`) and `provenance`
  (`doc_id`, `source_uri`, `source_version`, optional `span`).

### Interfaces
- FR-16: All interfaces call one internal service using the retrieval contract (design.md). Adapters only translate.
- FR-17: MCP server (FastMCP, streamable HTTP, auth) with `search_knowledge` — Phase 1.
- FR-18: MCP tools `get_concept`, `follow_links`, `get_document`, `list_sources` — Phase 4.
- FR-19: Native tool adapters for Claude (Anthropic SDK), OpenAI Agents SDK and Gemini, with identical
  tool names, descriptions and schemas — Phase 4.
- FR-20: One example agent per SDK answering the same question set (may build on `src/resource/base_*.py`).

### Evaluation
- FR-21: 50-question dev set + separate held-out test set, mixing single-fact, multi-hop and unanswerable questions.
- FR-22: recall@k / MRR script; answer grading (faithfulness, citation accuracy); per-agent-family breakdown.
- FR-23: Same-evidence check by replaying identical tool calls across adapters.
- FR-24: Failure suites: ambiguous, unanswerable, contradictory, stale, injected, unauthorized.

## Non-Functional Requirements
- NFR-01: Performance — p95 retrieval-API latency < 1.5 s at target concurrency (excluding generation);
  per-stage latency logged (embed, search, fuse, rerank).
- NFR-02: Security — tenant identity comes from credentials only; authorization enforced in the data-access
  layer on every search, graph hop, concept lookup and document fetch; zero cross-tenant results in tests.
- NFR-03: Security — retrieved content returned to agents as data with clear delimiters (prompt-injection defence).
- NFR-04: Determinism — same query + scope + index_version ⇒ same ranked evidence for every agent.
- NFR-05: Portability — `rag_core` has no agent-SDK imports.
- NFR-06: Observability — one structured log record per search call to CloudWatch
  (request_id, tenant_id, result IDs, per-stage latency; no raw document text).
- NFR-07: Cost — embedding + rerank + generation cost per query tracked; prompt caching for contextualization.
- NFR-08: Reproducibility — reproducible re-index script; embedding model/version stored per vector.

## Constraints
- Python throughout; AWS deployment (S3, Step Functions, Lambda, API Gateway, OpenSearch) via CDK or SAM.
- OKF is v0.1 — all OKF parsing lives in `rag_core/okf/`; nothing else reads frontmatter.
- Part-time pace (~8 weeks; 10–12 weeks if < 10 h/week). Re-plan at the week-3 gate.
- Tuning on the dev set only; held-out set runs only at phase gates.

## Out of Scope (v1)
- UI, multimodal documents (images, audio), fine-tuned embeddings.

## Success Criteria
- SC-01: Recall@5 ≥ 0.85 on the held-out set (metric definitions below).
- SC-02: The same question set answered by a Claude, an OpenAI and a Gemini agent through the same tools.
- SC-03: p95 retrieval-API latency < 1.5 s, excluding agent generation.
- SC-04: Every answer cites a concept or chunk ID that supports the claim; zero cross-tenant results in automated tests.

### Eval targets
| Metric | Target |
|--------|--------|
| Recall@5 | ≥ 0.85 |
| MRR | ≥ 0.70 |
| Faithfulness (LLM-graded, checked vs. 20-question human sample) | ≥ 0.90 |
| Citation accuracy (claim-level) | ≥ 0.95 |
| p95 latency (retrieval API only) | < 1.5 s |
| Cost per query | track only |

### Metric definitions (fixed in Phase 0)
- Relevant = contains text supporting the expected answer; a question may have several relevant documents.
- Recall@5 computed per question, then averaged; duplicate chunks from one document count once.
- Unanswerable questions scored on correct abstention, not recall.
- Citation accuracy is claim-level: the cited ID exists, was authorized for the caller, matches the indexed
  source version, and supports the claim.

## Phases and Gates
| Phase | Weeks | Scope | Exit gate |
|-------|-------|-------|-----------|
| 0 · Foundations | W1 | Corpus (30–100 docs), retrieval contract, provenance + authz rules, dev + held-out sets, metric definitions, repo/CI/config/secrets | Sets committed; empty pipeline runs end to end in CI |
| 1 · Raw-doc RAG baseline | W2–3 | FR-01–06, FR-12, FR-14–17, latency logging | Baseline recall@5 + MRR recorded; one agent answers via MCP with citations |
| 2 · Contextual retrieval | W4 | FR-07, re-index, A/B vs Phase 1 | Measured recall change; keep only if it improves |
| 3 · OKF knowledge layer | W5–6 | FR-08–11, FR-13, `okf-retrieve` prototype | Beats Phase 2 on dev set, especially multi-hop |
| 4 · Remaining interfaces | W7 | FR-18–20, cross-tenant tests per tool | All three agents get identical evidence for same query + scope |
| 5 · Evals + hardening | W8 | FR-22–24, load test, AWS deploy, regression gate, dashboards, write-up | Success criteria met or gaps documented |

Phase 3 carries the most schedule risk; if it slips, Phase 4 may start in parallel on the chunk index alone.

## Open Questions
- OQ-01: Which corpus — a real domain (product docs, dealership catalogues) or a public dataset?
- OQ-02: OpenSearch or pgvector?
- OQ-03: Own the OKF retriever or build on `okf-retrieve`? Decide after the Phase 3 prototype.
- OQ-04: Start date and weekly hours.
- OQ-05: Who reviews and verifies curated concepts?
- OQ-06: Gemini SDK — the existing `src/resource/base_gemini.py` uses `google-generativeai`; the plan names
  ADK / Gen AI SDK (`google-genai`). Migrate before Phase 4?
- Resolved: MCP server first (Phase 1), native adapters in Phase 4.
