"""IngestPipeline: orchestrates source read, chunk, embed and index upsert (idempotent)."""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from rag_core.envelope import validate_envelope_id
from rag_core.exceptions import ValidationError
from rag_core.models import Chunk, DeleteReport, IngestReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag_core.interfaces import Embedder, HybridIndex, SourceStore


class IngestPipeline:
    """Ingests and deletes documents; tenant always comes from the source manifest."""

    def __init__(
        self,
        sources: SourceStore,
        chunker: Callable[[str], list[str]],
        embedder: Embedder,
        index: HybridIndex,
        index_version: str,
    ) -> None:
        if not index_version.strip():
            raise ValidationError("index_version must not be empty")
        self._sources = sources
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._index_version = index_version

    def ingest(self, source_uri: str) -> IngestReport:
        """Index one source; skip it when its content hash is already indexed."""
        info = self._sources.describe_source(source_uri)
        raw = self._sources.read_source(source_uri)
        version = hashlib.sha256(raw).hexdigest()
        validate_envelope_id(info.doc_id)  # R-17: rejected at ingest
        pieces = self._chunker(raw.decode("utf-8"))
        report = IngestReport(doc_id=info.doc_id, index_version=self._index_version)
        indexed = self._index.document_version(info.tenant_id, info.doc_id, self._index_version)
        if indexed == version:
            report.chunks_skipped = len(pieces)
            return report
        # Embed first: a failure here must leave the previous chunks searchable (FR-03).
        chunks = self._build_chunks(info.tenant_id, info.doc_id, source_uri, version, pieces)
        if indexed is not None:
            self._index.delete_document(info.tenant_id, info.doc_id, self._index_version)
        self._index.upsert(chunks, self._index_version)
        report.chunks_written = len(chunks)
        return report

    def delete(self, doc_id: str) -> DeleteReport:
        """Remove every chunk of a document; the tenant is resolved from the manifest."""
        info = self._sources.describe_document(doc_id)
        removed = self._index.delete_document(info.tenant_id, info.doc_id, self._index_version)
        return DeleteReport(doc_id=doc_id, chunks_deleted=removed)

    def _build_chunks(
        self, tenant_id: str, doc_id: str, source_uri: str, version: str, pieces: list[str]
    ) -> list[Chunk]:
        vectors = self._embedder.embed(pieces) if pieces else []
        chunks: list[Chunk] = []
        cursor = 0
        for position, (text, vector) in enumerate(zip(pieces, vectors, strict=True)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{position}",
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    text=text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    span=(cursor, cursor + len(text)),
                    embedding_model=self._embedder.model_name,
                    embedding=tuple(vector),
                    metadata={"source_uri": source_uri, "source_version": version},
                )
            )
            cursor += len(text)
        return chunks
