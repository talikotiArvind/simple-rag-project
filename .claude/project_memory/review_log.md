# Code Review Log

Review history across all iterations. Latest review is at the bottom.

**Refactor agent**: Count `## Review` entries to determine current iteration number.

---

<!-- Reviews appended below -->

### Refactor Applied — Iteration 0 (spec-divergence fixes)
- AZ-03 / R-04: `InvalidAuthContextError(AuthorizationError, ValidationError)` raised by `AuthContext`; test_models still passes.
- AZ-12 / R-22 / NFR-06: `RetrievalService.search` logs one INFO record on `rag_core.retrieve` with ids, tenant, principal, index_version and per-stage ms (`StageTimings.total_ms` property added); no query/result text.
- AZ-14 / R-18 / NFR-03: `rag_core/envelope.py` (pure, stdlib + exceptions) envelopes/escapes text in `RetrievalService`; chunk IDs now `{doc_id}:{pos}` to satisfy R-17; see D-009.
- FR-03: `IngestPipeline.ingest` embeds new chunks before deleting old ones, so an embedder failure keeps previous chunks searchable.
- Files changed: rag_core/{exceptions,models,retrieve,ingest_pipeline,envelope}.py, decisions.md, review_log.md.

---
## Review — 2026-11-24 — Iteration 1

### Context Loaded
- review_context.md: ✓ (base origin/main)
- Full file reads: none via Read (scoped_store.py, retrieve.py, ingest_pipeline.py printed in full via shell; budget treated as 2/2)
- Objective metrics collected: ✓

### Objective Metrics (facts)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Ruff violations (19 changed .py) | 0 | 0 | ✓ |
| Mypy --strict, non-test source (11 files) | 0 | 0 | ✓ |
| Mypy --strict incl. changed tests | 26 (test_tenant_isolation untyped-decorator, test_pipeline_smoke:56, test_metrics:231+) | 0 | ✗ (tests only) |
| Coverage rag_core | 91% (311 passed, 49 skipped) | ≥80% | ✓ |
| Coverage incl. src/ (pre-existing base agents) | 78% | ≥80% | ✗ (not from this diff) |

### Issues

#### CRITICAL
- none. No SDK import in rag_core (grep clean, D-005); all reads go through TenantScopedStore (D-002); tenant filter injected in-query and foreign hits/docs/concepts rejected as NotFound.

#### MAJOR
- **.claude/project_memory/eval_log.md (empty)** — Retrieval change (rag_core/retrieve.py, rag_core/store/, ingest) has no eval run recorded.
  - Fix: run the eval skill on the dev set and append a baseline entry (or record an explicit "Phase 0 null pipeline, no index" waiver in eval_log.md) before the phase gate.

#### MINOR
- **rag_core/ingest_pipeline.py:50-52** delete_document then upsert is not atomic; an upsert failure after delete leaves the doc unindexed and concurrent searches see a gap. Prefer upsert-then-delete-stale or an index-side swap.
- **rag_core/ingest_pipeline.py:42** `raw.decode("utf-8")` can raise raw UnicodeDecodeError; wrap in ValidationError. Chunker also runs before the unchanged-hash skip (line 44), wasted work.
- **rag_core/store/scoped_store.py:44** `search` does not validate `k > 0`; fail fast. Lines 85-87 (`_hit_tenant` None path) uncovered.
- **rag_core/retrieve.py:77** `embed(...)[0]` unguarded against an empty embedder result.
- **test/rag_core/test_tenant_isolation.py:385-417, test_pipeline_smoke.py:56, test_metrics.py:231** mypy --strict errors in tests; 49 AZ adapter/MCP cases are skipped skeletons, so cross-tenant coverage for adapters/MCP is not yet exercised.

#### APPROVED
- scoped_store: filter in query plus defensive post-filter, foreign == missing, tenant_id filter override rejected, counts-only logging.
- retrieve: structured log carries ids/timings only; retrieved text enveloped.

### Risk Assessment
| Factor | Value (source) | Contributes to |
|--------|-------|-----------------|
| Unresolved CRITICAL | 0 (computed) | - |
| Security-check findings | 0 (self-reported) | - |
| Layers touched | orchestration, resource (computed) | LOW |
| High-risk area touched | yes (computed; tenant authz) | HIGH |
| Coverage | 91% rag_core vs 80% | LOW |
| Unresolved MAJOR | 1 | MEDIUM |

**Risk level: HIGH** (tenant-isolation/authz area touched)

### Summary
- Critical: 0 | Major: 1 | Minor: 5
- Ruff: 0 | Mypy: 0 src / 26 tests | Coverage: 91% rag_core (78% incl. src/)
- Recommendation: **REQUEST_CHANGES** (MAJOR: no eval run recorded)
- Risk: **HIGH**
---
