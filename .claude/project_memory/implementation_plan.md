# Implementation Plan — Phase 0: Foundations (Week 1)

Scope: ONLY Phase 0 of `requirements.md` "Phases and Gates". Later phases get no tasks.
Exit gate: dev + held-out sets committed; an empty pipeline runs end to end in CI (TASK-20).
Existing code: `src/resource/` (base agents + a2a) only. Tests live in `test/` (CLAUDE.md); `tests/` is legacy, untouched.

## Execution Strategy
- Order: human decisions (H) and scaffold (Phase 0.1) -> contract -> parallel groups A/B/C -> CI -> gate.
- `[BLOCKED-ON-USER]` tasks cannot be done by an agent. They run in parallel with the coding tasks; only
  TASK-18 (eval sets), TASK-19 (corpus) and TASK-20 (gate) depend on them. Everything else proceeds on
  synthetic fixtures, with no real corpus or secrets needed.
- Parallel-group tasks run in their own worktree and are merged one at a time, in Merge order, after every
  task in the group reports done. Shared files (`code_index.md`, `metrics.md`) merge per CLAUDE.md rules
  (index by file-path section; metrics counters summed). Source and test files below are unique per task.
- Assumptions to confirm: held-out set size >= 30 questions (plan only fixes dev = 50); Python 3.12 (matches CI);
  pytest + unittest per CLAUDE.md; contract uses stdlib dataclasses (design.md), no new pip deps in Phase 0.
- No `eval` skill exists in this repo's skill list; TASK-20 uses `evals/validate_sets.py` + the smoke pipeline
  and records the gate in `eval_log.md` by hand.

## Phase 0.0: Human decisions and inputs (BLOCKED-ON-USER, start immediately, parallel to everything)
- [ ] TASK-H1: Decide OQ-01 corpus (real domain vs public dataset), OQ-02 OpenSearch vs pgvector, OQ-04 start date and weekly hours  `[BLOCKED-ON-USER]`
  - Output: answers recorded as a `decisions.md` entry (D-009) and `requirements.md` open-question updates (system-designer/architect applies)
  - Unblocks: TASK-H2, TASK-H3, TASK-H4. Does NOT block TASK-01..17 (contract is store-agnostic).
  - Parallel group: PARALLEL-GROUP-NONE (human)
- [ ] TASK-H2: Gather 30-100 source docs (PDF/MD/HTML) into `corpus/` and write `corpus/manifest.json` (doc_id, source_uri, title, tenant_id, license) `[BLOCKED-ON-USER]`
  - Depends on: TASK-H1. Manifest must assign docs to >= 2 tenants (needed for cross-tenant tests) and include >= 1 doc updated twice (stale/version case). Check licences allow committing; else keep in S3 and commit manifest only.
  - Parallel group: PARALLEL-GROUP-NONE (human)
- [ ] TASK-H3: Author 50 dev questions with expected answers and relevant-doc labels `[BLOCKED-ON-USER]`
  - Depends on: TASK-H2, TASK-06 (schema). Mix: single-fact, multi-hop, unanswerable (target roughly 25/15/10, user may adjust). Relevance rule from requirements: relevant = contains text supporting the expected answer; several relevant docs allowed. Agents may draft candidates; the human confirms every label.
  - Output: `evals/dev_set.jsonl`
- [ ] TASK-H4: Author the separate held-out set, same mix, different questions, no tuning against it `[BLOCKED-ON-USER]`
  - Depends on: TASK-H2, TASK-06. Output: `evals/heldout_set.jsonl`. Preferably labelled without looking at dev-set results.
- [ ] TASK-H5: Secrets provisioning `[BLOCKED-ON-USER]`
  - User creates local `.env` from `.env.example` (OPENAI_API_KEY, ANTHROPIC_API_KEY, rerank key, AWS profile) and adds GitHub Actions secrets. Agents never read or enter real keys. Phase 0 CI must pass with NO secrets (smoke pipeline uses fakes). Real-key usage starts Phase 1.

## Phase 0.1: Scaffold (sequential)
- [x] TASK-01: Repo scaffold and tooling config
  - Files: `pyproject.toml` (pytest testpaths=test, ruff, mypy strict for `rag_core`, `evals`), empty-package `__init__.py` for `rag_core/`, `rag_core/store/`, `rag_core/ingest/`, `rag_core/okf/`, `rag_core/ranking/`, `adapters/`, `mcp_server/`, `evals/`, `tools/`, `test/rag_core/`; `knowledge/.gitkeep`, `infra/.gitkeep`, `examples/.gitkeep`
  - Build: layout from design.md "Repo Layout". Owns every `__init__.py` so later parallel tasks never touch them (parallelism rule). Must not alter `src/resource/` or existing `test/`.
  - Input: design.md. Output: tree above; `ruff` and `mypy` pass on empty packages.
  - Parallel group: PARALLEL-GROUP-NONE
  - Sub-agent: coding-agent

## Phase 0.2: Retrieval contract (sequential)
- [x] TASK-02: Contract models `rag_core/models.py`
  - Build: exactly the design.md contract: `AuthContext` (frozen), `SearchRequest` (validates on construction: non-empty query, 1<=top_k<=50, 0<=max_expansions<=10, filter keys from `ALLOWED_FILTER_KEYS` allow-list, `tenant_id` forbidden), `Provenance`, `SearchResult`, `SearchResponse`; plus `Chunk`, `Concept`, `Hit`, `Candidate`, `RankedLists`, `IngestReport`, `DeleteReport`, `StageTimings`, `DocumentView`, `SourceInfo`. `field(default_factory=...)` for mutables. Helper `sort_results()` implementing `(-score, result_id)` + dedupe by `result_id`.
  - Depends on: TASK-01. Output: `rag_core/models.py`. No I/O, no SDK imports.
  - Parallel group: PARALLEL-GROUP-NONE
  - Sub-agent: coding-agent

## Phase 0.3: Group A (after TASK-02; five independent files)
- [x] TASK-03: Protocols `rag_core/interfaces.py` <- PARALLEL-GROUP-A
  - Build: `Embedder`, `HybridIndex` (search/upsert/promote per design.md; every read takes `AuthContext`), `Reranker`, `SourceStore`, `ConceptRepository`, `Contextualizer` as `typing.Protocol`.
  - Depends on: TASK-02. Worktree: `.claude/worktrees/task-03/`  Merge order: 1
  - Sub-agent: coding-agent
- [x] TASK-04: Config and secrets handling `rag_core/config.py` <- PARALLEL-GROUP-A
  - Files: `rag_core/config.py`, `.env.example`, additions to `.gitignore` (`.env`, `*.pem`, `corpus/private/`)
  - Build: `Settings` typed dataclass loaded from env (python-dotenv already pinned): index name, embedding model, top_k defaults, `log_level`; secrets held as a `SecretStr`-style wrapper whose `repr` is redacted; fail-fast `ConfigError` for missing required values only when `require_secrets=True` (Phase 0 smoke runs with `False`). `.env.example` lists names only, no values. `.gitignore` is touched by this task only.
  - Depends on: TASK-02. Worktree: `.claude/worktrees/task-04/`  Merge order: 2
  - Sub-agent: coding-agent
- [x] TASK-05: D-005 import-boundary checker `tools/check_imports.py` <- PARALLEL-GROUP-A
  - Build: AST-based `ImportBoundaryChecker` (stdlib only, exit code 1 on violation, prints FILE:LINE). Rules: (1) anywhere under `rag_core/` forbid `agents`, `openai_agents`, `mcp`, `fastmcp`, `google.adk`, `src.resource` and `src/resource` equivalents, `adapters`, `mcp_server`; (2) provider SDKs (`openai`, `anthropic`, `cohere`, `voyageai`, `boto3`, `opensearchpy`, `google.generativeai`, `google.genai`) allowed ONLY in `rag_core/store/`, and must be function-level (lazy) imports there; (3) `rag_core/ingest|okf|ranking`, `retrieve.py`, `ingest_pipeline.py`, `models.py`, `interfaces.py` may import only stdlib, `rag_core.*`; flags relative-import and `importlib.import_module("...")` literal forms too. Rules in a data table at the top, so Phase 1 can extend them.
  - Depends on: TASK-01. Output: `tools/check_imports.py`
  - Worktree: `.claude/worktrees/task-05/`  Merge order: 3
  - Sub-agent: coding-agent
- [x] TASK-06: Eval schema and set loader `evals/schema.py` <- PARALLEL-GROUP-A
  - Build: `EvalQuestion` (question_id, question, qtype in {single_fact, multi_hop, unanswerable}, expected_answer, relevant_doc_ids: list[str] (empty iff unanswerable), tenant_id, notes) ; `EvalSet` with `load_jsonl(path)` that validates: unique ids, unanswerable has no relevant docs and answerable has >=1, multi-hop has >=2, doc ids exist in `corpus/manifest.json` (when supplied), no question text shared between dev and held-out sets (`assert_disjoint`).
  - Depends on: TASK-02. Output: `evals/schema.py`. Worktree: `.claude/worktrees/task-06/`  Merge order: 4
  - Sub-agent: coding-agent
- [x] TASK-07: Provenance and authorization rules doc `docs/authz_provenance.md` <- PARALLEL-GROUP-A
  - Build: normative rules derived from NFR-02/03, D-001..D-004, FR-15: identity only from credentials; tenant filter inside query; per-node authz on graph hops/concepts/documents; `source_version` = content hash; citation validity (4 conditions from metric definitions); delimiter format for returned text; log fields (no raw text). Includes a numbered "authorization test matrix" (surface x operation) that TASK-14 and later phases implement. Docs only; no code.
  - Depends on: TASK-02. Output: `docs/authz_provenance.md`. Worktree: `.claude/worktrees/task-07/`  Merge order: 5
  - Sub-agent: coding-agent (docs task)

## Phase 0.4: Group B (after Group A merged; depend on A outputs)
- [x] TASK-08: Tenant-scoped store skeleton `rag_core/store/scoped_store.py` <- PARALLEL-GROUP-B
  - Build: `TenantScopedStore` wrapping a `HybridIndex`, `SourceStore`, `ConceptRepository` (Protocols only, no SDK imports). Injects `tenant_id` filter into the index query (never post-filter), rejects any caller-supplied `tenant_id` filter, takes `AuthContext` on every method, raises `AuthorizationError` on missing/empty tenant. Concept/document fetch checks node tenant.
  - Depends on: TASK-03 (and TASK-07 for rules). Worktree: `.claude/worktrees/task-08/`  Merge order: 1
  - Sub-agent: coding-agent
- [x] TASK-09: Metrics `evals/metrics.py` <- PARALLEL-GROUP-B
  - Build: pure functions per the fixed definitions: `recall_at_k(ranked_doc_ids, relevant_doc_ids, k=5)` (per-question; chunks of one doc counted once, dedupe preserving rank), `mean_recall`, `reciprocal_rank`, `mrr`, `abstention_correct(results, threshold)` for unanswerable questions (scored separately, excluded from recall/MRR averages), `citation_valid(citation, ...)` stub encoding the 4 conditions with injected lookups. Breakdown by qtype.
  - Depends on: TASK-06. Output: `evals/metrics.py`. Worktree: `.claude/worktrees/task-09/`  Merge order: 2
  - Sub-agent: coding-agent
- [x] TASK-10: Set validator CLI `evals/validate_sets.py` <- PARALLEL-GROUP_B
  - Build: `python -m evals.validate_sets --dev evals/dev_set.jsonl --heldout evals/heldout_set.jsonl --manifest corpus/manifest.json`; enforces dev has exactly 50 questions, held-out >= 30 (configurable), type mix present in both (>=1 of each type), disjointness, schema validity; prints summary; exit 1 on failure. Also `--allow-missing` flag used only before TASK-H3/H4 land (CI gate step must NOT use it after TASK-18).
  - Depends on: TASK-06. Worktree: `.claude/worktrees/task-10/`  Merge order: 3
  - Sub-agent: coding-agent

## Phase 0.5: Empty pipeline (sequential)
- [x] TASK-11: Null pipeline and smoke runner
  - Files: `rag_core/retrieve.py` (`RetrievalService.search(auth, request)` orchestrating embed > scoped search > fuse > rerank via Protocols, records `StageTimings`, returns `SearchResponse` with `index_version`), `rag_core/ingest_pipeline.py` (`IngestPipeline.ingest/delete` orchestrating Protocols; idempotent on content hash), `evals/fakes.py` (in-memory `NullEmbedder`, `InMemoryHybridIndex` with tenant-tagged docs, `NullReranker`), `evals/run_pipeline_smoke.py` (ingest 2 tiny fixture docs for 2 tenants, run 1 search each, assert contract shape, determinism on repeat, no cross-tenant results; exit 0/1).
  - Depends on: TASK-04, TASK-08, TASK-09 (metrics used to compute recall on the fixture). Orchestration only: no ranking logic here (RRF/EvidenceScorer are Phase 1+; use identity fuse via Protocol default).
  - Parallel group: PARALLEL-GROUP-NONE
  - Sub-agent: coding-agent

## Phase 0.6: Tests (Group C, one unit-test agent per module; written to spec, blind maker-checker)
- [ ] TASK-12: `test/rag_core/test_models.py` <- PARALLEL-GROUP-C  (depends TASK-02; validation limits, allow-list, tenant_id forbidden, ordering/dedupe determinism)  Worktree: `.claude/worktrees/task-12/`  Merge order: 1  Sub-agent: unit-test
- [ ] TASK-13: `test/tools/test_check_imports.py` <- PARALLEL-GROUP-C  (depends TASK-05; temp-dir fixtures prove each D-005 rule fires and clean trees pass; also asserts the real repo's `rag_core/` is clean)  Worktree: `.claude/worktrees/task-13/`  Merge order: 2  Sub-agent: unit-test
- [ ] TASK-14: `test/rag_core/test_tenant_isolation.py` <- PARALLEL-GROUP-C  (depends TASK-08, TASK-07; tenant-isolation skeleton: base class `TenantIsolationContract` with one test per row of the authz matrix. Phase 0 enables search + document + concept rows against fakes; graph-hop, adapter and MCP rows are present as `@unittest.skip("Phase N: ...")` with reason, so later phases only unskip. Must assert zero cross-tenant results and that a caller-supplied `tenant_id` is rejected)  Worktree: `.claude/worktrees/task-14/`  Merge order: 3  Sub-agent: unit-test
- [ ] TASK-15: `test/evals/test_metrics.py` + `test/evals/test_schema.py` <- PARALLEL-GROUP-C  (depends TASK-06, TASK-09, TASK-10; hand-computed recall/MRR cases, duplicate-chunk-counts-once, unanswerable excluded, validator failure cases)  Worktree: `.claude/worktrees/task-15/`  Merge order: 4  Sub-agent: unit-test
- [ ] TASK-16: `test/rag_core/test_config.py` <- PARALLEL-GROUP-C  (depends TASK-04; secret repr redaction, missing-secret fail-fast, env isolation via patched `os.environ`)  Worktree: `.claude/worktrees/task-16/`  Merge order: 5  Sub-agent: unit-test
- [ ] TASK-17: `test/rag_core/test_pipeline_smoke.py` <- PARALLEL-GROUP-C  (depends TASK-11; idempotent re-ingest, stage timings present, `index_version` set, deterministic repeat, deletion removes chunks)  Worktree: `.claude/worktrees/task-17/`  Merge order: 6  Sub-agent: unit-test

## Phase 0.7: CI and gate (sequential)
- [ ] TASK-18: Commit eval sets `[BLOCKED-ON-USER]`
  - Depends on: TASK-H3, TASK-H4, TASK-10. Action: commit `evals/dev_set.jsonl` and `evals/heldout_set.jsonl`; `validate_sets` passes with no `--allow-missing`. (Human-authored; agent only runs validator and commits on user's request.)
- [ ] TASK-19: Commit corpus manifest and docs `[BLOCKED-ON-USER]`
  - Depends on: TASK-H2. Output: `corpus/manifest.json` (+ `corpus/` files or S3 URIs). Needed for validator doc-id checks.
- [ ] TASK-21: CI workflow `.github/workflows/ci.yml`
  - Build: triggers `push` to main + `pull_request`; Python 3.12; `pip install -r requirements.txt`; steps in order: (1) `python tools/check_imports.py` (D-005, blocking), (2) `ruff check rag_core evals tools adapters mcp_server`, (3) `mypy rag_core evals tools`, (4) `python -m evals.validate_sets ...` (blocking once TASK-18 lands; `--allow-missing` before), (5) `pytest test -q --cov=rag_core --cov=evals --cov-fail-under=80` (tenant-isolation tests included), (6) `python -m evals.run_pipeline_smoke` (the empty end-to-end pipeline). No secrets referenced anywhere. Does not modify the existing `python-checks.yml` / `unit-test.yml`.
  - Depends on: TASK-05, TASK-10, TASK-11, TASK-14, TASK-17. Parallel group: PARALLEL-GROUP-NONE. Sub-agent: coding-agent
- [ ] TASK-20: EXIT GATE: Phase 0
  - Check: (a) dev + held-out sets committed and `validate_sets` green; (b) CI run on a PR is green including D-005 check, tenant-isolation tests and smoke pipeline; (c) `docs/authz_provenance.md` and metric definitions present; (d) secrets absent from repo (`git grep` for key patterns, `bandit`, security-check skill). Record result in `eval_log.md` (dev-set run only; held-out set is NOT executed in Phase 0 beyond schema validation) and note the metrics-skill refresh in `metrics.md`.
  - Depends on: TASK-18, TASK-19, TASK-21, TASK-H5 (only for confirming local setup, CI needs none). Parallel group: PARALLEL-GROUP-NONE. Sub-agent: none (planner/user sign-off; code-reviewer for the PR)

## Progress Tracking
| Task | Status | Parallel Group | Notes |
|------|--------|---------------|-------|
| TASK-H1 | blocked-on-user | NONE | OQ-01, OQ-02, OQ-04 |
| TASK-H2 | blocked-on-user | NONE | needs H1 |
| TASK-H3 | blocked-on-user | NONE | needs H2, TASK-06 |
| TASK-H4 | blocked-on-user | NONE | needs H2, TASK-06 |
| TASK-H5 | blocked-on-user | NONE | secrets; not needed for Phase 0 CI |
| TASK-01 | done | NONE | |
| TASK-02 | done | NONE | |
| TASK-03 | done | A | |
| TASK-04 | done | A | |
| TASK-05 | done | A | |
| TASK-06 | done | A | |
| TASK-07 | done | A | |
| TASK-08 | done | B | feat/tenant-scoped-store |
| TASK-09 | done | B | |
| TASK-10 | done | B | feat/validate-sets |
| TASK-11 | done | NONE | feat/null-pipeline |
| TASK-12..17 | pending | C | |
| TASK-18 | blocked-on-user | NONE | |
| TASK-19 | blocked-on-user | NONE | |
| TASK-21 | pending | NONE | |
| TASK-20 | pending | NONE | exit gate |

## Parallelism Summary
| Group | Tasks | Can start after | Merge order |
|-------|-------|-----------------|-------------|
| PARALLEL-GROUP-NONE | TASK-01, TASK-02 | immediately (sequential) | n/a |
| PARALLEL-GROUP-A | TASK-03, 04, 05, 06, 07 | TASK-02 complete (TASK-05 needs only TASK-01) | 03, 04, 05, 06, 07 |
| PARALLEL-GROUP-B | TASK-08, 09, 10 | Group A merged | 08, 09, 10 |
| PARALLEL-GROUP-NONE | TASK-11 | Group B merged | n/a |
| PARALLEL-GROUP-C | TASK-12..17 | TASK-11 merged (12/13/15/16 could start earlier, but one group keeps one merge pass) | 12, 13, 14, 15, 16, 17 |
| PARALLEL-GROUP-NONE | TASK-21 then TASK-18/19 (user) then TASK-20 | Group C merged / user inputs | n/a |
| Human track | TASK-H1..H5 | immediately; H1 first | n/a |

Critical path: TASK-01 > 02 > 03 > 08 > 11 > 17 > 21 > 20. The user track (H1 > H2 > H3/H4 > TASK-18) is the real schedule risk: it should start on day 1.

## Risks
- Corpus choice (OQ-01) and label authoring are the longest-lead items and cannot be automated; labels must be human-confirmed or SC-01 is meaningless. Mitigation: start H1 today; agent drafts candidate questions only after H2.
- OQ-02 (OpenSearch vs pgvector) unresolved; mitigated because Phase 0 touches only Protocols and fakes. Must be decided before Phase 1.
- Held-out set leakage: committed to the repo, so discipline is process-only. Mitigation: never run it before gates; the smoke pipeline and tests use only fixture data.
- Existing `test/` vs `tests/` split: new tests go in `test/`; CI `pytest test` will not pick up `tests/`. Existing workflows are PR-only and diff-scoped; new `ci.yml` is separate to avoid regressions.
- `pyproject.toml` (TASK-01) sets mypy strict only for new packages so the existing `src/resource` code does not start failing.
- TASK-05 import checker can miss dynamic imports; acceptable for Phase 0, noted for review.
- Part-time pace: Phase 0 is ~1 week only if H-track inputs arrive by mid-week; otherwise TASK-18/20 slip while code tasks finish early.
