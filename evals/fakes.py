"""In-memory fakes of the rag_core Protocols for the null pipeline and tests (no I/O)."""
from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

from rag_core.exceptions import NotFoundError, ValidationError
from rag_core.models import (
    Candidate,
    Concept,
    DocumentView,
    Hit,
    Provenance,
    RankedLists,
    SourceInfo,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rag_core.models import AuthContext, Chunk

EMBED_DIM = 32
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


class NullEmbedder:
    """Deterministic bag-of-words hashing embedder; similar texts get similar vectors."""

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model."""
        return "null-hash-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalised vector per text."""
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for token in tokenize(text):
            slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
            vec[slot % EMBED_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class NullReranker:
    """Identity reranker: keeps the incoming order, truncated to top_k."""

    def rerank(
        self, auth: AuthContext, query: str, candidates: Sequence[Candidate], top_k: int
    ) -> list[Candidate]:
        """Return the first top_k candidates unchanged."""
        return list(candidates)[:top_k]


class InMemoryHybridIndex:
    """Versioned in-memory index; the tenant filter is applied inside the query."""

    def __init__(self) -> None:
        self._versions: dict[str, dict[str, Chunk]] = {}
        self._live: str | None = None

    @property
    def live_version(self) -> str | None:
        """Currently promoted index version, if any."""
        return self._live

    def upsert(self, chunks: Sequence[Chunk], index_version: str) -> None:
        """Write chunks (keyed by tenant and chunk_id) into the version."""
        store = self._versions.setdefault(index_version, {})
        for chunk in chunks:
            store[f"{chunk.tenant_id}/{chunk.chunk_id}"] = chunk

    def promote(self, index_version: str) -> None:
        """Make index_version the live version."""
        if index_version not in self._versions:
            raise NotFoundError(f"unknown index version: {index_version}")
        self._live = index_version

    def document_version(self, tenant_id: str, doc_id: str, index_version: str) -> str | None:
        """Return the source_version of an indexed document, or None."""
        for chunk in self._chunks(index_version):
            if chunk.tenant_id == tenant_id and chunk.doc_id == doc_id:
                return chunk.metadata.get("source_version")
        return None

    def delete_document(self, tenant_id: str, doc_id: str, index_version: str) -> int:
        """Remove a document's chunks from the version; return the count removed."""
        store = self._versions.get(index_version, {})
        doomed = [k for k, c in store.items() if c.tenant_id == tenant_id and c.doc_id == doc_id]
        for key in doomed:
            del store[key]
        return len(doomed)

    def count(self, index_version: str, tenant_id: str | None = None) -> int:
        """Number of chunks in a version, optionally for one tenant."""
        return sum(1 for c in self._chunks(index_version) if tenant_id in (None, c.tenant_id))

    def search(
        self,
        auth: AuthContext,
        query_vector: list[float],
        query_text: str,
        k: int,
        filters: Mapping[str, str],
    ) -> RankedLists:
        """Return BM25-like and cosine lists restricted to the filtered tenant."""
        if k <= 0:
            raise ValidationError("k must be positive")
        tenant_id = filters.get("tenant_id", "")
        pool = [c for c in self._chunks(self._live) if c.tenant_id == tenant_id]
        terms = set(tokenize(query_text))
        bm25 = [(float(len(terms & set(tokenize(c.text)))), c) for c in pool]
        knn = [(_dot(query_vector, c.embedding), c) for c in pool]
        return RankedLists(bm25=_top(bm25, k), knn=_top(knn, k))

    def _chunks(self, index_version: str | None) -> list[Chunk]:
        if index_version is None:
            return []
        return list(self._versions.get(index_version, {}).values())


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _top(scored: list[tuple[float, Chunk]], k: int) -> list[Hit]:
    positive = [(s, c) for s, c in scored if s > 0.0]
    positive.sort(key=lambda sc: (-sc[0], sc[1].chunk_id))
    return [Hit(result_id=c.chunk_id, score=s, chunk=c) for s, c in positive[:k]]


class InMemorySourceStore:
    """In-memory manifest plus raw documents."""

    def __init__(self) -> None:
        self._infos: dict[str, SourceInfo] = {}
        self._raw: dict[str, bytes] = {}

    def add(self, info: SourceInfo, raw: bytes) -> None:
        """Register or replace a source (keyed by source_uri)."""
        self._infos[info.source_uri] = info
        self._raw[info.source_uri] = raw

    def read_source(self, source_uri: str) -> bytes:
        """Return raw bytes of a source."""
        if source_uri not in self._raw:
            raise NotFoundError("source not found")
        return self._raw[source_uri]

    def describe_source(self, source_uri: str) -> SourceInfo:
        """Return manifest metadata for a source_uri."""
        if source_uri not in self._infos:
            raise NotFoundError("source not found")
        return self._infos[source_uri]

    def describe_document(self, doc_id: str) -> SourceInfo:
        """Return manifest metadata for a doc_id."""
        for info in self._infos.values():
            if info.doc_id == doc_id:
                return info
        raise NotFoundError("document not found")

    def list_sources(self, auth: AuthContext) -> list[SourceInfo]:
        """Return the caller tenant's sources ordered by doc_id."""
        own = [i for i in self._infos.values() if i.tenant_id == auth.tenant_id]
        return sorted(own, key=lambda i: i.doc_id)

    def get_document(self, auth: AuthContext, doc_id: str) -> DocumentView:
        """Return a document owned by the caller's tenant."""
        info = self.describe_document(doc_id)
        if info.tenant_id != auth.tenant_id:
            raise NotFoundError("document not found")
        raw = self._raw[info.source_uri]
        prov = Provenance(
            doc_id=doc_id,
            source_uri=info.source_uri,
            source_version=hashlib.sha256(raw).hexdigest(),
        )
        return DocumentView(
            doc_id=doc_id, text=raw.decode("utf-8"), provenance=prov, tenant_id=info.tenant_id
        )


class InMemoryConceptRepository:
    """Empty concept repository for Phase 0 (no OKF content yet)."""

    def get_concept(self, auth: AuthContext, concept_id: str) -> Concept:
        """Always raises: there are no concepts."""
        raise NotFoundError("concept not found")

    def neighbours(
        self, auth: AuthContext, concept_id: str, edge_types: frozenset[str]
    ) -> list[Concept]:
        """Always empty."""
        return []
