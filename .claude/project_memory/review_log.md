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
