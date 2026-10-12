# Architecture Decisions Log

Append decisions here. Do not edit existing entries — only add new ones.

Format:
```
## [DATE] — [Decision Title]
**Decision**: What was decided
**Context**: Why this decision was needed
**Rationale**: Why this choice over alternatives
**Consequences**: What this enables and what it constrains
```

---

<!-- Decisions appended below -->

## 2026-10-12 — D-001 Canonical retrieval contract
**Decision**: One internal `RetrievalService` with the dataclass contract in design.md; MCP server and all SDK adapters only translate.
**Context**: Technical review flagged divergent per-SDK tool behaviour as the top risk to "same evidence for every agent".
**Rationale**: A single contract makes same-evidence testable by replaying identical calls.
**Consequences**: Contract changes need a new decision entry; adapters cannot add filtering or ranking.

## 2026-10-12 — D-002 Authorization in the data-access layer
**Decision**: `TenantScopedStore` wraps every store; `AuthContext` built from credentials at the API surface and passed down on every call. Tenant filter is applied inside the index query.
**Context**: Graph expansion and `get_document` can reach nodes the direct search never returned.
**Rationale**: Enforcing at the lowest layer means a new tool or code path cannot skip it.
**Rejected**: Handler-level checks (easy to forget per tool); post-filtering hits (leaks via counts/scores, starves top_k).
**Consequences**: Every store method signature takes `AuthContext`; cross-tenant test required per tool.

## 2026-10-12 — D-003 Evidence-aware ranking, not verified-first
**Decision**: `EvidenceScorer` weighs relevance, verification, freshness and source authority; verified never overrides.
**Context**: Verified-first let general or stale concepts outrank more specific chunk evidence.
**Rejected**: Hard verified-first ordering.
**Consequences**: Weights are tuned on the dev set only and recorded in eval_log.md.

## 2026-10-12 — D-004 Graph expansion is additive and bounded
**Decision**: 1 hop, allowed edge types, per-node authz, ≤ `max_expansions`, `via="graph"`, never displaces top_k direct hits.
**Rationale**: Keeps direct-search recall measurable and bounds latency and leak surface.

## 2026-10-12 — D-005 "No agent-SDK imports in rag_core" scoped precisely
**Decision**: `rag_core` must not import agent/tool-calling frameworks (`agents`, ADK, MCP, `src.resource.*`). Model-provider clients (OpenAI embeddings, Anthropic Haiku, rerank APIs) are allowed only inside `rag_core/store/`, behind `interfaces.py` Protocols, with lazy imports.
**Context**: The plan's embedder (OpenAI) and contextualizer (Haiku) are provider SDK calls that live in rag_core.
**Consequences**: Service and orchestration modules import only Protocols; enforce with a CI import check (Phase 0).

## 2026-10-12 — D-006 Repo layout: plan layout + explicit layer packages
**Decision**: Keep the plan's top-level packages; inside `rag_core` add `store/` (data access) and `ranking/` (service), keep `retrieve.py` and add `ingest_pipeline.py` as orchestration.
**Rejected**: The generic `src/api|domain|infrastructure` template from system-designer.md (conflicts with the plan's shareable monorepo layout).
**Consequences**: `prep_review_context.py` layer markers updated to these names.

## 2026-10-12 — D-007 Versioned index with alias promotion
**Decision**: Each (re)index writes a new physical index; an alias is promoted after evals pass; rollback = promote previous.
**Consequences**: `index_version` in every response and eval record; stale-chunk cleanup is per version.

## 2026-10-12 — D-008 MCP first, native adapters in Phase 4
**Decision**: Resolved open question — MCP server with `search_knowledge` in Phase 1; Claude/OpenAI/Gemini adapters in Phase 4.
**Rationale**: One working interface by week 3 de-risks the contract before tripling the surface.
