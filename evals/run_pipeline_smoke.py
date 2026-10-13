"""Null end-to-end pipeline smoke runner: `python -m evals.run_pipeline_smoke` (exit 0/1)."""
from __future__ import annotations

import sys
from dataclasses import dataclass

from evals.fakes import (
    InMemoryConceptRepository,
    InMemoryHybridIndex,
    InMemorySourceStore,
    NullEmbedder,
    NullReranker,
)
from evals.metrics import recall_at_k
from rag_core.ingest_pipeline import IngestPipeline
from rag_core.models import AuthContext, SearchRequest, SearchResponse, SourceInfo
from rag_core.retrieve import RetrievalService
from rag_core.store.scoped_store import TenantScopedStore

INDEX_VERSION = "v1"
FIXTURES = (
    ("acme", "doc-a", "mem://a", "Refunds are issued within thirty days of purchase."),
    ("globex", "doc-g", "mem://g", "Satellite antennas require quarterly calibration."),
)
CHECK_NAMES = (
    "ingest", "idempotency", "contract", "determinism", "isolation", "recall", "delete"
)
QUERIES = {"acme": ("refunds purchase days", "doc-a"), "globex": ("antennas calibration", "doc-g")}


def sentence_chunker(text: str) -> list[str]:
    """Split text into sentences, keeping the trailing period."""
    return [s.strip() + "." for s in text.split(".") if s.strip()]


@dataclass
class Wiring:
    """Assembled null pipeline."""

    index: InMemoryHybridIndex
    ingest: IngestPipeline
    retrieval: RetrievalService


def build_wiring() -> Wiring:
    """Create fakes and the services under test."""
    sources = InMemorySourceStore()
    for tenant, doc_id, uri, text in FIXTURES:
        sources.add(SourceInfo(doc_id=doc_id, source_uri=uri, tenant_id=tenant), text.encode())
    index, embedder = InMemoryHybridIndex(), NullEmbedder()
    store = TenantScopedStore(index, sources, InMemoryConceptRepository())
    return Wiring(
        index=index,
        ingest=IngestPipeline(sources, sentence_chunker, embedder, index, INDEX_VERSION),
        retrieval=RetrievalService(embedder, store, NullReranker(), INDEX_VERSION),
    )


def _search(wiring: Wiring, tenant: str) -> SearchResponse:
    auth = AuthContext(tenant_id=tenant, principal="smoke")
    return wiring.retrieval.search(auth, SearchRequest(query=QUERIES[tenant][0], top_k=3))


def _check_contract(tenant: str, response: SearchResponse) -> list[str]:
    errors: list[str] = []
    if response.index_version != INDEX_VERSION:
        errors.append(f"{tenant}: index_version missing")
    if not response.results:
        return [*errors, f"{tenant}: no results"]
    for result in response.results:
        if not result.result_id or not result.provenance:
            errors.append(f"{tenant}: result lacks id or provenance")
    own_doc = QUERIES[tenant][1]
    if any(p.doc_id != own_doc for r in response.results for p in r.provenance):
        errors.append(f"{tenant}: cross-tenant result")
    if response.timings.search_ms < 0.0:
        errors.append(f"{tenant}: stage timings missing")
    return errors


def _ids(response: SearchResponse) -> list[str]:
    return [r.result_id for r in response.results]


def run_checks(wiring: Wiring) -> tuple[list[str], list[str]]:
    """Run all smoke checks; return (errors, passed-check names)."""
    errors: list[str] = []
    first = [wiring.ingest.ingest(uri) for _, _, uri, _ in FIXTURES]
    wiring.index.promote(INDEX_VERSION)
    if not all(r.chunks_written > 0 for r in first):
        errors.append("initial ingest wrote no chunks")
    again = [wiring.ingest.ingest(uri) for _, _, uri, _ in FIXTURES]
    if any(r.chunks_written != 0 or r.chunks_skipped == 0 for r in again):
        errors.append("re-ingest not idempotent")
    for tenant in QUERIES:
        response = _search(wiring, tenant)
        errors += _check_contract(tenant, response)
        if _ids(response) != _ids(_search(wiring, tenant)):
            errors.append(f"{tenant}: non-deterministic results")
        doc_ids = [p.doc_id for r in response.results for p in r.provenance]
        if recall_at_k(doc_ids, [QUERIES[tenant][1]]) != 1.0:
            errors.append(f"{tenant}: recall@k below 1.0 on fixture")
    errors += _check_delete(wiring)
    return errors, list(CHECK_NAMES)


def _check_delete(wiring: Wiring) -> list[str]:
    report = wiring.ingest.delete("doc-a")
    left = wiring.index.count(INDEX_VERSION, "acme")
    if report.chunks_deleted == 0 or left != 0:
        return ["delete did not remove chunks"]
    if not wiring.index.count(INDEX_VERSION, "globex"):
        return ["delete removed another tenant's chunks"]
    return []


def main() -> int:
    """Run the smoke checks and print a short summary."""
    errors, names = run_checks(build_wiring())
    if errors:
        print("FAIL: " + "; ".join(errors))
        return 1
    print(f"PASS: {len(names)} checks ({', '.join(names)}); recall@5=1.00 on fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
