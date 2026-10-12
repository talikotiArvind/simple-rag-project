"""RetrievalService: orchestrates embed -> scoped search -> fuse -> rerank via Protocols only."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from rag_core.exceptions import ValidationError
from rag_core.models import (
    Candidate,
    Provenance,
    SearchResponse,
    SearchResult,
    StageTimings,
    sort_results,
)

if TYPE_CHECKING:
    from rag_core.interfaces import Embedder, Reranker
    from rag_core.models import AuthContext, Hit, RankedLists, SearchRequest
    from rag_core.store.scoped_store import TenantScopedStore


class Fuser(Protocol):
    """Merges the BM25 and k-NN lists into one candidate list."""

    def fuse(self, lists: RankedLists, top_k: int) -> list[Candidate]:
        """Return at most top_k de-duplicated candidates, best first."""
        ...


class InterleaveFuser:
    """Phase 0 placeholder: alternates BM25 and k-NN hits (no score fusion)."""

    def fuse(self, lists: RankedLists, top_k: int) -> list[Candidate]:
        """Alternate lists, dedupe by result_id, score by position so order is stable."""
        merged: dict[str, Hit] = {}
        longest = max(len(lists.bm25), len(lists.knn))
        for pos in range(longest):
            for source in (lists.bm25, lists.knn):
                if pos < len(source):
                    merged.setdefault(source[pos].result_id, source[pos])
        ordered = list(merged.values())[:top_k]
        return [
            Candidate(hit=replace(hit, score=1.0 / (rank + 1))) for rank, hit in enumerate(ordered)
        ]


class RetrievalService:
    """Runs one tenant-scoped search and shapes the response."""

    def __init__(
        self,
        embedder: Embedder,
        store: TenantScopedStore,
        reranker: Reranker,
        index_version: str,
        fuser: Fuser | None = None,
    ) -> None:
        if not index_version.strip():
            raise ValidationError("index_version must not be empty")
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._index_version = index_version
        self._fuser: Fuser = fuser or InterleaveFuser()

    def search(self, auth: AuthContext, request: SearchRequest) -> SearchResponse:
        """Embed, search within the caller's tenant, fuse, rerank and return ordered results."""
        timings = StageTimings()
        start = time.perf_counter()
        vector = self._embedder.embed([request.query])[0]
        timings.embed_ms = _elapsed_ms(start)

        start = time.perf_counter()
        filters = {k: str(v) for k, v in request.filters.items()}
        lists = self._store.search(auth, vector, request.query, request.top_k, filters)
        timings.search_ms = _elapsed_ms(start)

        start = time.perf_counter()
        fused = self._fuser.fuse(lists, request.top_k)
        timings.fuse_ms = _elapsed_ms(start)

        # Phase 0: expand_graph is accepted but graph expansion is not implemented (expand_ms=0).
        start = time.perf_counter()
        ranked = self._reranker.rerank(auth, request.query, fused, request.top_k)
        timings.rerank_ms = _elapsed_ms(start)

        results = sort_results(self._to_result(c) for c in ranked)[: request.top_k]
        return SearchResponse(
            results=results,
            request_id=uuid.uuid4().hex,
            index_version=self._index_version,
            timings=timings,
        )

    @staticmethod
    def _to_result(candidate: Candidate) -> SearchResult:
        hit = candidate.hit
        if hit.chunk is not None:
            meta = hit.chunk.metadata
            prov = Provenance(
                doc_id=hit.chunk.doc_id,
                source_uri=meta.get("source_uri", ""),
                source_version=meta.get("source_version", hit.chunk.content_hash),
                span=hit.chunk.span,
            )
            return SearchResult(
                result_id=hit.result_id,
                result_type="chunk",
                text=hit.chunk.text,
                score=hit.score,
                provenance=[prov],
                via=candidate.via,
            )
        concept = hit.concept
        return SearchResult(
            result_id=hit.result_id,
            result_type="concept",
            text="; ".join(concept.claims) if concept else "",
            score=hit.score,
            provenance=list(concept.sources) if concept else [],
            verified=concept.verified if concept else False,
            via=candidate.via,
        )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
