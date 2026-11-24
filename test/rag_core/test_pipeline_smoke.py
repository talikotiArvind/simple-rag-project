"""Pipeline smoke tests for rag_core/ingest_pipeline.py, rag_core/retrieve.py and evals runner.

Expected behaviour is derived from requirements.md (FR-03, FR-04, FR-05, FR-12, FR-15, NFR-01,
NFR-04) and design.md (IngestPipeline / RetrievalService contracts), not from the implementation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import unittest
from unittest.mock import patch

from evals.fakes import (
    InMemoryConceptRepository,
    InMemoryHybridIndex,
    InMemorySourceStore,
    NullEmbedder,
    NullReranker,
)
from evals.run_pipeline_smoke import main as smoke_main
from evals.run_pipeline_smoke import sentence_chunker
from rag_core.exceptions import NotFoundError, ValidationError
from rag_core.ingest_pipeline import IngestPipeline
from rag_core.models import AuthContext, SearchRequest, SearchResponse, SourceInfo
from rag_core.retrieve import RetrievalService
from rag_core.store.scoped_store import TenantScopedStore

VERSION = "v1"
DOC_A = "Alpha refunds arrive quickly. Beta shipping takes days. Gamma support is friendly."
DOC_G = "Alpha antennas need calibration. Delta satellites orbit slowly."


class PipelineTestBase(unittest.TestCase):
    """Builds a fresh in-memory pipeline with two tenants for every test."""

    def setUp(self) -> None:
        self.sources = InMemorySourceStore()
        self.index = InMemoryHybridIndex()
        self.ingest_embedder = NullEmbedder()
        self.embedder = NullEmbedder()
        store = TenantScopedStore(self.index, self.sources, InMemoryConceptRepository())
        self.pipeline = IngestPipeline(
            self.sources, sentence_chunker, self.ingest_embedder, self.index, VERSION
        )
        self.service = RetrievalService(self.embedder, store, NullReranker(), VERSION)
        self.acme = AuthContext(tenant_id="acme", principal="tester")
        self.globex = AuthContext(tenant_id="globex", principal="tester")
        self._add("acme", "doc-a", "mem://a", DOC_A)
        self._add("globex", "doc-g", "mem://g", DOC_G)

    def _add(self, tenant: str, doc_id: str, uri: str, text: str) -> None:
        info = SourceInfo(doc_id=doc_id, source_uri=uri, tenant_id=tenant)
        self.sources.add(info, text.encode("utf-8"))

    def _ingest(self, uri: str):  # noqa: ANN202
        report = self.pipeline.ingest(uri)
        if self.index.live_version is None:
            self.index.promote(VERSION)
        return report

    def _search(self, auth: AuthContext, query: str, **kwargs: object) -> SearchResponse:
        return self.service.search(auth, SearchRequest(query=query, **kwargs))  # type: ignore[arg-type]

    @staticmethod
    def _ids(response: SearchResponse) -> list[str]:
        return [r.result_id for r in response.results]


class TestIngest(PipelineTestBase):
    """FR-03 / FR-04: stable ids, provenance, idempotency, replacement."""

    def test_ingest_writes_one_chunk_per_piece(self) -> None:
        report = self._ingest("mem://a")
        self.assertEqual(report.chunks_written, 3)
        self.assertEqual(report.doc_id, "doc-a")
        self.assertEqual(report.index_version, VERSION)

    def test_chunk_ids_are_stable_across_independent_pipelines(self) -> None:
        self._ingest("mem://a")
        first = self._ids(self._search(self.acme, "alpha beta gamma refunds shipping support"))
        other = PipelineTestBase()
        other.setUp()
        other._ingest("mem://a")
        second = self._ids(other._search(other.acme, "alpha beta gamma refunds shipping support"))
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_results_carry_provenance_with_content_hash_version(self) -> None:
        self._ingest("mem://a")
        response = self._search(self.acme, "alpha refunds")
        expected = hashlib.sha256(DOC_A.encode("utf-8")).hexdigest()
        self.assertTrue(response.results)
        for result in response.results:
            prov = result.provenance[0]
            self.assertEqual(prov.doc_id, "doc-a")
            self.assertEqual(prov.source_uri, "mem://a")
            self.assertEqual(prov.source_version, expected)

    def test_reingest_unchanged_is_idempotent_and_reported_skipped(self) -> None:
        self._ingest("mem://a")
        count_before = self.index.count(VERSION)
        report = self._ingest("mem://a")
        self.assertEqual(report.chunks_written, 0)
        self.assertEqual(report.chunks_skipped, 3)
        self.assertEqual(self.index.count(VERSION), count_before)

    def test_reingest_unchanged_does_not_duplicate_search_results(self) -> None:
        self._ingest("mem://a")
        before = self._ids(self._search(self.acme, "alpha beta gamma"))
        self._ingest("mem://a")
        self.assertEqual(self._ids(self._search(self.acme, "alpha beta gamma")), before)

    def test_changed_source_replaces_chunks_and_no_stale_chunk_is_searchable(self) -> None:
        self._ingest("mem://a")
        self._add("acme", "doc-a", "mem://a", "Omega warranty lasts years.")
        report = self._ingest("mem://a")
        self.assertEqual(report.chunks_written, 1)
        self.assertEqual(self.index.count(VERSION, "acme"), 1)
        self.assertEqual(self._search(self.acme, "alpha beta gamma refunds").results, [])
        self.assertEqual(len(self._search(self.acme, "omega warranty").results), 1)

    def test_changed_source_updates_source_version(self) -> None:
        self._ingest("mem://a")
        new_text = "Omega warranty lasts years."
        self._add("acme", "doc-a", "mem://a", new_text)
        self._ingest("mem://a")
        prov = self._search(self.acme, "omega").results[0].provenance[0]
        self.assertEqual(prov.source_version, hashlib.sha256(new_text.encode()).hexdigest())

    def test_unknown_source_raises_not_found(self) -> None:
        with self.assertRaises(NotFoundError):
            self.pipeline.ingest("mem://missing")


class TestIngestFailure(PipelineTestBase):
    """FR-03: ingest is retry-safe; a failing source must not corrupt the index."""

    def test_source_read_failure_leaves_previous_chunks_searchable(self) -> None:
        self._ingest("mem://a")
        before = self._ids(self._search(self.acme, "alpha beta gamma"))
        with patch.object(self.sources, "read_source", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.pipeline.ingest("mem://a")
        self.assertEqual(self._ids(self._search(self.acme, "alpha beta gamma")), before)
        self.assertEqual(self.index.count(VERSION, "acme"), 3)

    def test_source_failure_for_one_doc_does_not_affect_other_doc(self) -> None:
        self._ingest("mem://g")
        with patch.object(self.sources, "read_source", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                self.pipeline.ingest("mem://a")
        self.assertTrue(self._search(self.globex, "antennas").results)

    def test_embedder_failure_on_changed_source_keeps_previous_chunks(self) -> None:
        """Spec FR-03 (retry-safe): a failed re-ingest must not drop the old, valid chunks."""
        self._ingest("mem://a")
        self._add("acme", "doc-a", "mem://a", "Omega warranty lasts years.")
        with patch.object(self.ingest_embedder, "embed", side_effect=RuntimeError("embed down")):
            with self.assertRaises(RuntimeError):
                self.pipeline.ingest("mem://a")
        self.assertTrue(self._search(self.acme, "alpha beta gamma refunds").results)


class TestDelete(PipelineTestBase):
    """FR-04: deletion removes every chunk of the document."""

    def test_delete_removes_all_chunks_and_reports_count(self) -> None:
        self._ingest("mem://a")
        report = self.pipeline.delete("doc-a")
        self.assertEqual(report.doc_id, "doc-a")
        self.assertEqual(report.chunks_deleted, 3)
        self.assertEqual(self.index.count(VERSION, "acme"), 0)

    def test_deleted_document_is_no_longer_searchable(self) -> None:
        self._ingest("mem://a")
        self.assertTrue(self._search(self.acme, "alpha").results)
        self.pipeline.delete("doc-a")
        self.assertEqual(self._search(self.acme, "alpha").results, [])

    def test_delete_leaves_other_documents_intact(self) -> None:
        self._ingest("mem://a")
        self._ingest("mem://g")
        self.pipeline.delete("doc-a")
        self.assertTrue(self._search(self.globex, "antennas").results)

    def test_delete_unknown_doc_raises_clear_not_found_error(self) -> None:
        # Spec only says delete "removes stale chunks"; behaviour for an unknown doc is not
        # specified. A clear NotFoundError (no crash, no silent success) is asserted.
        with self.assertRaises(NotFoundError):
            self.pipeline.delete("no-such-doc")

    def test_delete_registered_but_never_indexed_doc_reports_zero(self) -> None:
        report = self.pipeline.delete("doc-a")
        self.assertEqual(report.chunks_deleted, 0)

    def test_reingest_after_delete_restores_chunks(self) -> None:
        self._ingest("mem://a")
        self.pipeline.delete("doc-a")
        report = self._ingest("mem://a")
        self.assertEqual(report.chunks_written, 3)


class TestSearch(PipelineTestBase):
    """FR-05 / FR-12 / FR-15 / NFR-01 / NFR-04."""

    def setUp(self) -> None:
        super().setUp()
        self._ingest("mem://a")
        self._ingest("mem://g")

    def test_response_reports_index_version(self) -> None:
        self.assertEqual(self._search(self.acme, "alpha").index_version, VERSION)

    def test_response_has_request_id(self) -> None:
        self.assertTrue(self._search(self.acme, "alpha").request_id)

    def test_results_have_result_id_and_non_empty_provenance(self) -> None:
        response = self._search(self.acme, "alpha beta gamma")
        self.assertTrue(response.results)
        for result in response.results:
            self.assertTrue(result.result_id)
            self.assertTrue(result.provenance)

    def test_at_most_top_k_results(self) -> None:
        response = self._search(self.acme, "alpha beta gamma refunds shipping support", top_k=2)
        self.assertEqual(len(response.results), 2)

    def test_top_k_one_returns_single_result(self) -> None:
        self.assertEqual(len(self._search(self.acme, "alpha beta gamma", top_k=1).results), 1)

    def test_results_sorted_by_descending_score_then_result_id(self) -> None:
        results = self._search(self.acme, "alpha beta gamma refunds shipping").results
        keys = [(-r.score, r.result_id) for r in results]
        self.assertGreater(len(keys), 1)
        self.assertEqual(keys, sorted(keys))

    def test_stage_timings_present_and_non_negative(self) -> None:
        timings = self._search(self.acme, "alpha").timings
        for name in ("embed_ms", "search_ms", "fuse_ms", "rerank_ms"):
            value = getattr(timings, name)
            self.assertIsInstance(value, float, name)
            self.assertGreaterEqual(value, 0.0, name)

    def test_repeated_search_gives_identical_ordered_result_ids(self) -> None:
        query = "alpha beta gamma refunds shipping support"
        first = self._ids(self._search(self.acme, query))
        for _ in range(3):
            self.assertEqual(self._ids(self._search(self.acme, query)), first)

    def test_no_cross_tenant_results(self) -> None:
        for auth, doc in ((self.acme, "doc-a"), (self.globex, "doc-g")):
            response = self._search(auth, "alpha antennas refunds satellites calibration")
            self.assertTrue(response.results)
            docs = {p.doc_id for r in response.results for p in r.provenance}
            self.assertEqual(docs, {doc})

    def test_expand_graph_without_concepts_adds_nothing(self) -> None:
        with_graph = self._search(self.acme, "alpha beta gamma", expand_graph=True)
        without = self._search(self.acme, "alpha beta gamma", expand_graph=False)
        self.assertEqual(self._ids(with_graph), self._ids(without))
        self.assertTrue(all(r.via != "graph" for r in with_graph.results))

    def test_query_without_matches_returns_empty_results(self) -> None:
        response = self._search(self.acme, "zzzunmatched")
        self.assertEqual(response.results, [])
        self.assertEqual(response.index_version, VERSION)


class TestConstructorValidation(PipelineTestBase):
    """Fail-fast validation of index_version."""

    def test_retrieval_service_rejects_empty_index_version(self) -> None:
        store = TenantScopedStore(self.index, self.sources, InMemoryConceptRepository())
        for bad in ("", "   "):
            with self.assertRaises(ValidationError):
                RetrievalService(self.embedder, store, NullReranker(), bad)

    def test_ingest_pipeline_rejects_empty_index_version(self) -> None:
        for bad in ("", "   "):
            with self.assertRaises(ValidationError):
                IngestPipeline(self.sources, sentence_chunker, self.embedder, self.index, bad)


class TestSmokeRunner(unittest.TestCase):
    """`python -m evals.run_pipeline_smoke` entry point."""

    def test_main_returns_zero_and_prints_pass(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = smoke_main()
        self.assertEqual(code, 0)
        self.assertTrue(buffer.getvalue().startswith("PASS"))


if __name__ == "__main__":
    unittest.main()
