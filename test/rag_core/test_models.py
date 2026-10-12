"""Tests for rag_core/models.py, written to the Retrieval Contract in design.md."""
from __future__ import annotations

import dataclasses
import unittest
from typing import Any

from rag_core.exceptions import ValidationError
from rag_core.models import (
    ALLOWED_FILTER_KEYS,
    AuthContext,
    Provenance,
    SearchRequest,
    SearchResponse,
    SearchResult,
    sort_results,
)


def _result(result_id: str, score: float, **kwargs: Any) -> SearchResult:
    return SearchResult(
        result_id=result_id, result_type="chunk", text=f"t-{result_id}", score=score, **kwargs
    )


class TestAuthContext(unittest.TestCase):
    def test_holds_given_fields(self) -> None:
        auth = AuthContext(tenant_id="t1", principal="alice", scopes=frozenset({"read"}))
        self.assertEqual(
            (auth.tenant_id, auth.principal, auth.scopes), ("t1", "alice", frozenset({"read"}))
        )

    def test_is_frozen(self) -> None:
        auth = AuthContext(tenant_id="t1", principal="alice", scopes=frozenset())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            auth.tenant_id = "t2"  # type: ignore[misc]

    def test_scopes_cannot_be_mutated(self) -> None:
        auth = AuthContext(tenant_id="t1", principal="alice", scopes=frozenset({"a"}))
        self.assertFalse(hasattr(auth.scopes, "add"))

    def test_is_hashable_and_equal_by_value(self) -> None:
        a = AuthContext("t1", "alice", frozenset({"x"}))
        b = AuthContext("t1", "alice", frozenset({"x"}))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_empty_tenant_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AuthContext(tenant_id="", principal="alice", scopes=frozenset())

    def test_whitespace_tenant_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AuthContext(tenant_id="   ", principal="alice", scopes=frozenset())

    def test_empty_principal_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AuthContext(tenant_id="t1", principal="", scopes=frozenset())

    def test_whitespace_principal_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AuthContext(tenant_id="t1", principal=" \t", scopes=frozenset())


class TestSearchRequestDefaults(unittest.TestCase):
    def test_defaults(self) -> None:
        req = SearchRequest(query="hello")
        self.assertEqual(req.top_k, 5)
        self.assertEqual(req.filters, {})
        self.assertTrue(req.expand_graph)
        self.assertEqual(req.max_expansions, 3)

    def test_filters_not_shared_between_instances(self) -> None:
        a = SearchRequest(query="a")
        b = SearchRequest(query="b")
        a.filters["source"] = "x"
        self.assertEqual(b.filters, {})
        self.assertIsNot(a.filters, b.filters)


class TestSearchRequestQuery(unittest.TestCase):
    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="")

    def test_whitespace_query_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="  \n\t ")

    def test_query_with_surrounding_whitespace_accepted(self) -> None:
        self.assertEqual(SearchRequest(query=" hi ").query, " hi ")


class TestSearchRequestTopK(unittest.TestCase):
    def test_boundaries_accepted(self) -> None:
        self.assertEqual(SearchRequest(query="q", top_k=1).top_k, 1)
        self.assertEqual(SearchRequest(query="q", top_k=50).top_k, 50)

    def test_below_range_rejected(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SearchRequest(query="q", top_k=value)

    def test_above_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", top_k=51)


class TestSearchRequestMaxExpansions(unittest.TestCase):
    def test_boundaries_accepted(self) -> None:
        self.assertEqual(SearchRequest(query="q", max_expansions=0).max_expansions, 0)
        self.assertEqual(SearchRequest(query="q", max_expansions=10).max_expansions, 10)

    def test_negative_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", max_expansions=-1)

    def test_above_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", max_expansions=11)


class TestSearchRequestFilters(unittest.TestCase):
    def test_each_allowed_key_accepted(self) -> None:
        self.assertTrue(ALLOWED_FILTER_KEYS)
        for key in sorted(ALLOWED_FILTER_KEYS):
            with self.subTest(key=key):
                req = SearchRequest(query="q", filters={key: "v"})
                self.assertEqual(req.filters, {key: "v"})

    def test_all_allowed_keys_together_accepted(self) -> None:
        filters = {key: "v" for key in ALLOWED_FILTER_KEYS}
        self.assertEqual(SearchRequest(query="q", filters=filters).filters, filters)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", filters={"bogus": "v"})

    def test_unknown_key_alongside_allowed_rejected(self) -> None:
        key = sorted(ALLOWED_FILTER_KEYS)[0]
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", filters={key: "v", "bogus": "v"})

    def test_tenant_id_key_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "tenant_id"):
            SearchRequest(query="q", filters={"tenant_id": "other"})

    def test_tenant_id_not_in_allow_list(self) -> None:
        self.assertNotIn("tenant_id", ALLOWED_FILTER_KEYS)

    def test_tenant_id_rejected_even_with_allowed_keys(self) -> None:
        key = sorted(ALLOWED_FILTER_KEYS)[0]
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", filters={key: "v", "tenant_id": "x"})


class TestProvenance(unittest.TestCase):
    def test_span_defaults_to_none(self) -> None:
        prov = Provenance(doc_id="d", source_uri="u", source_version="h")
        self.assertIsNone(prov.span)

    def test_span_can_be_set(self) -> None:
        prov = Provenance(doc_id="d", source_uri="u", source_version="h", span=(1, 4))
        self.assertEqual(prov.span, (1, 4))


class TestSearchResultAndResponse(unittest.TestCase):
    def test_result_defaults(self) -> None:
        res = _result("r1", 1.0)
        self.assertEqual(res.via, "search")
        self.assertFalse(res.verified)
        self.assertEqual(res.provenance, [])

    def test_result_provenance_not_shared(self) -> None:
        a, b = _result("a", 1.0), _result("b", 1.0)
        a.provenance.append(Provenance("d", "u", "h"))
        self.assertEqual(b.provenance, [])
        self.assertIsNot(a.provenance, b.provenance)

    def test_response_holds_fields(self) -> None:
        results = [_result("a", 1.0)]
        resp = SearchResponse(results=results, request_id="req", index_version="v1")
        self.assertEqual(
            (resp.results, resp.request_id, resp.index_version), (results, "req", "v1")
        )


class TestSortResults(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(sort_results([]), [])

    def test_orders_by_score_descending(self) -> None:
        out = sort_results([_result("a", 0.1), _result("b", 0.9), _result("c", 0.5)])
        self.assertEqual([r.result_id for r in out], ["b", "c", "a"])

    def test_ties_broken_by_result_id_ascending(self) -> None:
        out = sort_results([_result("c", 0.5), _result("a", 0.5), _result("b", 0.5)])
        self.assertEqual([r.result_id for r in out], ["a", "b", "c"])

    def test_negative_scores_ordered_descending(self) -> None:
        out = sort_results([_result("a", -2.0), _result("b", -1.0)])
        self.assertEqual([r.result_id for r in out], ["b", "a"])

    def test_dedupe_keeps_best_score(self) -> None:
        low, high = _result("a", 0.2), _result("a", 0.8)
        out = sort_results([low, high])
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], high)

    def test_dedupe_keeps_best_score_regardless_of_order(self) -> None:
        low, high = _result("a", 0.2), _result("a", 0.8)
        out = sort_results([high, low])
        self.assertEqual([r.score for r in out], [0.8])

    def test_dedupe_does_not_affect_other_ids(self) -> None:
        out = sort_results([_result("a", 0.2), _result("a", 0.8), _result("b", 0.5)])
        self.assertEqual([(r.result_id, r.score) for r in out], [("a", 0.8), ("b", 0.5)])

    def test_input_order_independence(self) -> None:
        items = [
            _result("a", 0.5),
            _result("b", 0.5),
            _result("c", 0.9),
            _result("a", 0.1),
            _result("d", 0.2),
        ]
        expected = [(r.result_id, r.score) for r in sort_results(items)]
        for perm in (list(reversed(items)), items[2:] + items[:2], items[1::2] + items[::2]):
            with self.subTest(order=[r.text for r in perm]):
                self.assertEqual([(r.result_id, r.score) for r in sort_results(perm)], expected)

    def test_does_not_mutate_input(self) -> None:
        items = [_result("b", 0.1), _result("a", 0.9), _result("a", 0.3)]
        snapshot = list(items)
        sort_results(items)
        self.assertEqual(len(items), len(snapshot))
        self.assertTrue(all(x is y for x, y in zip(items, snapshot, strict=True)))

    def test_accepts_iterator(self) -> None:
        out = sort_results(iter([_result("b", 0.1), _result("a", 0.2)]))
        self.assertEqual([r.result_id for r in out], ["a", "b"])

    def test_returns_new_list(self) -> None:
        items = [_result("a", 1.0)]
        self.assertIsNot(sort_results(items), items)

    def test_repeated_calls_are_deterministic(self) -> None:
        items = [_result("b", 0.5), _result("a", 0.5)]
        self.assertEqual(sort_results(items), sort_results(items))


if __name__ == "__main__":
    unittest.main()
