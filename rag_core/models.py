"""Provider-neutral retrieval contract and pipeline data models (no I/O)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from rag_core.exceptions import ValidationError

ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {"doc_type", "source", "language", "updated_after"}
)
MIN_TOP_K = 1
MAX_TOP_K = 50
MAX_EXPANSIONS_LIMIT = 10


@dataclass(frozen=True)
class AuthContext:
    """Caller identity, built from credentials and never from tool arguments."""

    tenant_id: str
    principal: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty")
        if not self.principal.strip():
            raise ValidationError("principal must not be empty")


@dataclass
class SearchRequest:
    """Validated search request; filters are restricted to an allow-list."""

    query: str
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    expand_graph: bool = True
    max_expansions: int = 3

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValidationError("query must not be empty")
        if not MIN_TOP_K <= self.top_k <= MAX_TOP_K:
            raise ValidationError(f"top_k must be in [{MIN_TOP_K}, {MAX_TOP_K}], got {self.top_k}")
        if not 0 <= self.max_expansions <= MAX_EXPANSIONS_LIMIT:
            raise ValidationError(
                f"max_expansions must be in [0, {MAX_EXPANSIONS_LIMIT}], got {self.max_expansions}"
            )
        self._validate_filters()

    def _validate_filters(self) -> None:
        if "tenant_id" in self.filters:
            raise ValidationError("tenant_id filter is forbidden; tenant comes from AuthContext")
        unknown = sorted(set(self.filters) - ALLOWED_FILTER_KEYS)
        if unknown:
            raise ValidationError(f"unsupported filter keys: {unknown}")


@dataclass
class Provenance:
    """Where a result's content came from."""

    doc_id: str
    source_uri: str
    source_version: str
    span: tuple[int, int] | None = None


@dataclass
class SearchResult:
    """One retrieved chunk or concept with provenance."""

    result_id: str
    result_type: Literal["chunk", "concept"]
    text: str
    score: float
    provenance: list[Provenance] = field(default_factory=list)
    verified: bool = False
    via: Literal["search", "graph"] = "search"


@dataclass
class StageTimings:
    """Per-request stage latencies in milliseconds."""

    embed_ms: float = 0.0
    search_ms: float = 0.0
    fuse_ms: float = 0.0
    expand_ms: float = 0.0
    rerank_ms: float = 0.0


@dataclass
class SearchResponse:
    """Ordered results for one request."""

    results: list[SearchResult]
    request_id: str
    index_version: str
    timings: StageTimings = field(default_factory=StageTimings)


@dataclass(frozen=True)
class Chunk:
    """An indexable text chunk of a source document."""

    chunk_id: str
    doc_id: str
    tenant_id: str
    text: str
    content_hash: str
    context: str = ""
    span: tuple[int, int] | None = None
    embedding_model: str = ""
    embedding: tuple[float, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Concept:
    """An OKF concept: claims with sources, verification flag and typed edges."""

    concept_id: str
    title: str
    claims: list[str] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    verified: bool = False
    edges: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Hit:
    """Raw ranked hit returned by the store."""

    result_id: str
    score: float
    chunk: Chunk | None = None
    concept: Concept | None = None


@dataclass
class Candidate:
    """A hit plus the ranking features used by the scorer."""

    hit: Hit
    features: dict[str, float] = field(default_factory=dict)
    via: Literal["search", "graph"] = "search"


@dataclass
class RankedLists:
    """BM25 and k-NN ranked lists from a hybrid index."""

    bm25: list[Hit] = field(default_factory=list)
    knn: list[Hit] = field(default_factory=list)


@dataclass
class IngestReport:
    """Outcome of ingesting one source."""

    doc_id: str
    chunks_written: int = 0
    chunks_skipped: int = 0
    failures: list[str] = field(default_factory=list)
    index_version: str = ""


@dataclass
class DeleteReport:
    """Outcome of deleting one document."""

    doc_id: str
    chunks_deleted: int = 0


@dataclass
class DocumentView:
    """Full document text with provenance."""

    doc_id: str
    text: str
    provenance: Provenance


@dataclass(frozen=True)
class SourceInfo:
    """Listing entry for an available source."""

    doc_id: str
    source_uri: str
    title: str = ""


def sort_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Dedupe by result_id keeping the best score, ordered by (-score, result_id)."""
    best: dict[str, SearchResult] = {}
    for result in results:
        current = best.get(result.result_id)
        if current is None or result.score > current.score:
            best[result.result_id] = result
    return sorted(best.values(), key=lambda r: (-r.score, r.result_id))
