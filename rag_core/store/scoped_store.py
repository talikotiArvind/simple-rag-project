"""TenantScopedStore: the only read path; injects the tenant filter and checks per-node tenancy."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.exceptions import AuthorizationError, NotFoundError
from rag_core.models import Hit, RankedLists

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rag_core.interfaces import ConceptRepository, HybridIndex, SourceStore
    from rag_core.models import AuthContext, Concept, DocumentView, SourceInfo

logger = logging.getLogger(__name__)

TENANT_FILTER_KEY = "tenant_id"


class TenantScopedStore:
    """Decorator over index, source store and concept repository enforcing tenant isolation."""

    def __init__(
        self, index: HybridIndex, sources: SourceStore, concepts: ConceptRepository
    ) -> None:
        self._index = index
        self._sources = sources
        self._concepts = concepts

    def search(
        self,
        auth: AuthContext,
        query_vector: list[float],
        query_text: str,
        k: int,
        filters: Mapping[str, str],
    ) -> RankedLists:
        """Query with the tenant filter inside the query; drop foreign hits defensively."""
        self._require_tenant(auth)
        if TENANT_FILTER_KEY in filters:
            raise AuthorizationError("tenant_id filter is forbidden; tenant comes from AuthContext")
        scoped = {**filters, TENANT_FILTER_KEY: auth.tenant_id}
        raw = self._index.search(auth, query_vector, query_text, k, scoped)
        return RankedLists(
            bm25=self._keep_own_hits(auth, raw.bm25), knn=self._keep_own_hits(auth, raw.knn)
        )

    def get_document(self, auth: AuthContext, doc_id: str) -> DocumentView:
        """Return the document, or NotFoundError if missing or owned by another tenant."""
        self._require_tenant(auth)
        try:
            doc = self._sources.get_document(auth, doc_id)
        except (AuthorizationError, NotFoundError):
            raise NotFoundError("document not found") from None
        if doc.tenant_id != auth.tenant_id:
            raise NotFoundError("document not found")
        return doc

    def list_sources(self, auth: AuthContext) -> list[SourceInfo]:
        """Return only the caller tenant's sources ordered by doc_id."""
        self._require_tenant(auth)
        listed = self._sources.list_sources(auth)
        own = [s for s in listed if s.tenant_id == auth.tenant_id]
        if len(own) != len(listed):
            logger.error("source store returned sources outside tenant scope")
        return sorted(own, key=lambda s: s.doc_id)

    def get_concept(self, auth: AuthContext, concept_id: str) -> Concept:
        """Return the concept, or NotFoundError if missing or owned by another tenant."""
        self._require_tenant(auth)
        try:
            concept = self._concepts.get_concept(auth, concept_id)
        except (AuthorizationError, NotFoundError):
            raise NotFoundError("concept not found") from None
        if concept.tenant_id != auth.tenant_id:
            raise NotFoundError("concept not found")
        return concept

    def neighbours(
        self, auth: AuthContext, concept_id: str, edge_types: frozenset[str]
    ) -> list[Concept]:
        """Return linked concepts; other tenants' nodes are dropped silently (no backfill)."""
        self.get_concept(auth, concept_id)
        linked = self._concepts.neighbours(auth, concept_id, edge_types)
        own = [c for c in linked if c.tenant_id == auth.tenant_id]
        return sorted(own, key=lambda c: c.concept_id)

    @staticmethod
    def _require_tenant(auth: AuthContext) -> None:
        if not auth.tenant_id or not auth.tenant_id.strip():
            raise AuthorizationError("tenant_id must not be empty")

    @staticmethod
    def _hit_tenant(hit: Hit) -> str | None:
        if hit.chunk is not None:
            return hit.chunk.tenant_id
        if hit.concept is not None:
            return hit.concept.tenant_id
        return None

    def _keep_own_hits(self, auth: AuthContext, hits: list[Hit]) -> list[Hit]:
        own = [h for h in hits if self._hit_tenant(h) == auth.tenant_id]
        if len(own) != len(hits):
            # Index filter bug: counts only, never ids or text of foreign data
            logger.error("index returned %d hit(s) outside tenant scope", len(hits) - len(own))
        return own
