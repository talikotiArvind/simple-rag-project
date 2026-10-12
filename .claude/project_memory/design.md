# System Design — Unified Retrieval Layer

Seeded from the 2026-10-12 project plan. `system-designer` refines this per phase; the retrieval contract
below is fixed in Phase 0 and changes only through a `decisions.md` entry.

## Architecture Overview
One provider-neutral library, `rag_core`, owns ingest, indexing, OKF loading, hybrid search, graph expansion
and ranking. Every agent-facing surface — the MCP server, the three native tool adapters, the Lambda handler —
is a thin translator from its own wire format to the retrieval contract and back. Authorization is enforced in
the data-access layer through a tenant-scoped store wrapper, so no higher layer can forget it. Ranking is
evidence-aware (verification is one feature among relevance, freshness and authority), and every result
carries provenance so all three agent families can cite identically.

## Repo Layout (from the plan; `store/` and `ranking/` added to make layers explicit)
```
rag_core/                    # provider-neutral; no agent-SDK imports
├── models.py                # contract: AuthContext, SearchRequest, SearchResult, Provenance, SearchResponse, Concept, Chunk
├── interfaces.py            # Protocols: Embedder, HybridIndex, Reranker, SourceStore, ConceptRepository, Contextualizer
├── store/                   # DATA ACCESS — only place with I/O
│   ├── opensearch_index.py  # HybridIndex impl (BM25 + k-NN, versioned aliases)
│   ├── s3_source.py         # SourceStore impl
│   ├── openai_embedder.py   # Embedder impl (lazy SDK import)
│   ├── rerank_client.py     # Reranker impl (Cohere/Voyage; local cross-encoder fallback)
│   ├── haiku_contextualizer.py  # Contextualizer impl (prompt-cached)
│   ├── concept_repo.py      # ConceptRepository over the OKF bundle index
│   └── scoped_store.py      # TenantScopedStore: injects tenant filter, per-node authz — wraps all of the above
├── ingest/                  # SERVICE — parse, chunk, hash, build Chunk models (no I/O)
├── okf/                     # SERVICE — bundle parsing, validation, link graph (only reader of OKF frontmatter)
├── ranking/                 # SERVICE — RRF fusion, evidence-aware scorer, graph-expansion policy, dedupe
├── ingest_pipeline.py       # ORCHESTRATION — IngestPipeline (parse > chunk > contextualize > embed > upsert)
└── retrieve.py              # ORCHESTRATION — RetrievalService (search > expand > merge > rerank)
adapters/                    # API SURFACE — claude_tools.py, openai_tools.py, gemini_tools.py
mcp_server/                  # API SURFACE — FastMCP app, auth.py (credentials → AuthContext)
knowledge/                   # OKF bundle (index.md + concept files)
examples/                    # one agent per SDK (may subclass src/resource/base_*.py)
evals/                       # dev set, held-out set, recall@k/MRR, answer grading, replay harness
infra/                       # CDK or SAM
src/resource/                # existing provider base agents + a2a tool adapter (agent side, not rag_core)
```

## Layer Responsibilities
### Data access — `rag_core/store/`
- Purpose: all network/storage I/O: OpenSearch, S3, embedding API, rerank API, Haiku.
- Key classes: `OpenSearchHybridIndex`, `S3SourceStore`, `OpenAIEmbedder`, `RerankClient`,
  `HaikuContextualizer`, `ConceptRepository`, `TenantScopedStore`.
- Rule: no ranking or business logic. Every read path goes through `TenantScopedStore`, which takes an
  `AuthContext` on every call and applies the tenant filter **inside** the query, not after.

### Service logic — `rag_core/ingest/`, `rag_core/okf/`, `rag_core/ranking/`
- Purpose: pure transformations.
- Key classes: `DocumentParser` (PDF/MD/HTML), `HeadingChunker`, `ContentHasher`, `OkfBundleLoader`,
  `OkfValidator`, `LinkGraph`, `RrfFuser`, `EvidenceScorer`, `GraphExpansionPolicy`, `ResultDeduper`.
- Rule: no I/O, no SDK imports; inputs and outputs are contract models.

### Orchestration — `rag_core/retrieve.py`, `rag_core/ingest_pipeline.py`
- Purpose: sequence store and service calls; record per-stage latency.
- Key classes: `RetrievalService`, `IngestPipeline`.
- Rule: no data transformation of its own.

### API surface — `mcp_server/`, `adapters/`, `infra/` handlers
- Purpose: build `AuthContext` from credentials, translate tool args ↔ `SearchRequest`/`SearchResponse`,
  wrap result text in delimiters for the model.
- Key classes: `McpApp`, `CredentialAuthenticator`, `ClaudeToolAdapter`, `OpenAIToolAdapter`, `GeminiToolAdapter`.
- Rule: no ranking, no filtering, no authz decisions beyond resolving identity.

## Retrieval Contract (fixed in Phase 0)
```python
@dataclass(frozen=True)
class AuthContext:            # built from credentials, never from tool arguments
    tenant_id: str
    principal: str
    scopes: frozenset[str]

@dataclass
class SearchRequest:
    query: str
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    expand_graph: bool = True
    max_expansions: int = 3

@dataclass
class Provenance:
    doc_id: str
    source_uri: str
    source_version: str        # content hash of the indexed source
    span: tuple[int, int] | None = None

@dataclass
class SearchResult:
    result_id: str
    result_type: Literal["chunk", "concept"]
    text: str
    score: float
    provenance: list[Provenance]   # a concept lists the sources behind its claims
    verified: bool = False
    via: Literal["search", "graph"] = "search"

@dataclass
class SearchResponse:
    results: list[SearchResult]
    request_id: str
    index_version: str
```
Contract rules:
- `SearchRequest` validates on construction: non-empty query, `1 <= top_k <= 50`, `0 <= max_expansions <= 10`,
  `filters` keys from an allow-list (never `tenant_id`).
- Graph results are labelled `via="graph"`, appended after the top_k direct hits, at most `max_expansions`.
- Ordering is deterministic: sort by `(-score, result_id)`; dedupe by `result_id`, and chunks of one document
  are deduped per the recall definition only in evals, not in responses.

## Interface Contracts
```python
class RetrievalService:
    def search(self, auth: AuthContext, request: SearchRequest) -> SearchResponse: ...
    def get_concept(self, auth: AuthContext, concept_id: str) -> Concept: ...
    def follow_links(self, auth: AuthContext, concept_id: str, edge_types: frozenset[str]) -> list[SearchResult]: ...
    def get_document(self, auth: AuthContext, doc_id: str) -> DocumentView: ...
    def list_sources(self, auth: AuthContext) -> list[SourceInfo]: ...

class IngestPipeline:
    def ingest(self, source_uri: str) -> IngestReport: ...          # idempotent on content hash
    def delete(self, doc_id: str) -> DeleteReport: ...              # removes stale chunks

class HybridIndex(Protocol):
    def search(self, auth: AuthContext, query_vector: list[float], query_text: str, k: int,
               filters: Mapping[str, str]) -> RankedLists: ...   # BM25 list + k-NN list, tenant-filtered
    def upsert(self, chunks: Sequence[Chunk], index_version: str) -> None: ...
    def promote(self, index_version: str) -> None: ...             # alias swap; rollback = promote(previous)

class RrfFuser:
    def fuse(self, rankings: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]: ...
class EvidenceScorer:
    def rank(self, candidates: Sequence[Candidate], top_k: int) -> list[SearchResult]: ...
class GraphExpansionPolicy:
    def select(self, seeds: Sequence[SearchResult], neighbours: Sequence[Candidate],
               max_expansions: int) -> list[SearchResult]: ...
```

## Typed Data Models
| Model | Layer | Purpose |
|-------|-------|---------|
| `AuthContext` | crosses all | Caller identity from credentials |
| `SearchRequest` / `SearchResponse` / `SearchResult` / `Provenance` | contract | Retrieval contract |
| `Chunk` | service → store | chunk_id, doc_id, text, context, hash, span, embedding_model |
| `Concept` | service → store | concept_id, claims with sources, verified, edges |
| `Hit` / `Candidate` | store → service | raw ranked hit; hit + ranking features |
| `IngestReport` / `DeleteReport` | orchestration | counts, skipped (unchanged hash), failures |
| `StageTimings` | orchestration | embed/search/fuse/expand/rerank ms per request |

## Design Patterns Applied
| Pattern | Where | Why |
|---------|-------|-----|
| Protocol + adapter | `interfaces.py` / `store/` | Swap OpenSearch↔pgvector, Cohere↔local reranker without touching services |
| Decorator | `TenantScopedStore` | Authz cannot be bypassed by a new caller |
| Strategy | `EvidenceScorer` weights, fusion | A/B ranking variants against the dev set |
| Adapter | `adapters/`, `mcp_server/` | One contract, four wire formats |
| Blue/green alias | index versions | Rollback = alias swap |

## Abstraction Reuse
| New | Reuses | Reason |
|-----|--------|--------|
| `adapters/*_tools.py` | pattern of `src/resource/a2a/tool_adapter.py` (`to_anthropic_tools`, `to_gemini_tools`, `to_openai_handoffs`) | Same per-SDK tool translation problem |
| `examples/*` agents | `src/resource/base_claude.py`, `base_openai.py`, `base_gemini.py` | Already wrap each SDK |
| everything in `rag_core` | — | New abstraction required: no retrieval code exists |

## External Dependencies (added in the phase that needs them)
| Library | Phase | Purpose |
|---------|-------|---------|
| `opensearch-py` | 1 | BM25 + k-NN in one index |
| `openai` | 1 | text-embedding-3-large |
| `fastmcp` | 1 | MCP server, streamable HTTP |
| `pypdf`, `beautifulsoup4`, `markdown-it-py` | 1 | Parsers |
| `boto3` | 1 | S3 source store |
| `cohere` or `voyageai` | 1 | Rerank API |
| `anthropic` | 2 | Haiku contextualizer (prompt caching) |
| `okf-retrieve` | 3 | Prototype only; decision OQ-03 |
| `aws-cdk-lib` or SAM | 1–5 | Infra |

## Architecture Decisions
See `decisions.md` (D-001 … D-008).
