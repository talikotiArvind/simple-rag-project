"""Provider-neutral Protocols implemented by store/ and adapter layers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rag_core.models import (
        AuthContext,
        Candidate,
        Chunk,
        Concept,
        DocumentView,
        RankedLists,
        SourceInfo,
    )


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors; the model name is recorded on every chunk."""

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in input order."""
        ...


@runtime_checkable
class HybridIndex(Protocol):
    """BM25 + k-NN index; reads are always tenant-filtered."""

    def search(
        self,
        auth: AuthContext,
        query_vector: list[float],
        query_text: str,
        k: int,
        filters: Mapping[str, str],
    ) -> RankedLists:
        """Return the BM25 list and the k-NN list for the caller's tenant."""
        ...

    def upsert(self, chunks: Sequence[Chunk], index_version: str) -> None:
        """Write chunks into the given index version."""
        ...

    def promote(self, index_version: str) -> None:
        """Swap the live alias to index_version; rollback is promote(previous)."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates by relevance to the query."""

    def rerank(
        self, auth: AuthContext, query: str, candidates: Sequence[Candidate], top_k: int
    ) -> list[Candidate]:
        """Return at most top_k candidates, best first."""
        ...


@runtime_checkable
class SourceStore(Protocol):
    """Reads original documents and lists available sources."""

    def get_document(self, auth: AuthContext, doc_id: str) -> DocumentView:
        """Return the full document text with provenance."""
        ...

    def list_sources(self, auth: AuthContext) -> list[SourceInfo]:
        """Return the caller's sources ordered by doc_id."""
        ...

    def read_source(self, source_uri: str) -> bytes:
        """Read raw bytes for ingestion (trusted pipeline path, no caller identity)."""
        ...


@runtime_checkable
class ConceptRepository(Protocol):
    """Access to OKF concepts and their typed edges."""

    def get_concept(self, auth: AuthContext, concept_id: str) -> Concept:
        """Return one concept; the node's tenant must match the caller's."""
        ...

    def neighbours(
        self, auth: AuthContext, concept_id: str, edge_types: frozenset[str]
    ) -> list[Concept]:
        """Return concepts linked by the given edge types, ordered by concept_id."""
        ...


@runtime_checkable
class Contextualizer(Protocol):
    """Produces the situating context prepended to a chunk before embedding."""

    def contextualize(self, document_text: str, chunk_text: str) -> str:
        """Return a short context string for chunk_text within its document."""
        ...
