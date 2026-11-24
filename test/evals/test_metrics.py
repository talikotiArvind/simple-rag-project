"""Tests for evals/metrics.py (expected values hand-computed from requirements.md definitions)."""

from __future__ import annotations

import unittest

from evals.metrics import (
    CitationCondition,
    QuestionResult,
    abstention_accuracy,
    abstention_correct,
    citation_valid,
    mean_recall,
    mrr,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from evals.schema import EvalQuestion, QuestionType
from rag_core.exceptions import ValidationError


class TestRecallAtK(unittest.TestCase):
    """Per-question recall@k."""

    def test_relevant_at_rank_5_is_found(self) -> None:
        self.assertEqual(recall_at_k(["d1", "d2", "d3", "d4", "d5", "d6"], {"d5"}, k=5), 1.0)

    def test_relevant_at_rank_6_is_missed(self) -> None:
        self.assertEqual(recall_at_k(["d1", "d2", "d3", "d4", "d5", "d6"], {"d6"}, k=5), 0.0)

    def test_default_k_is_five(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c", "d", "e", "f"], {"f"}), 0.0)

    def test_partial_recall_for_multiple_relevant_docs(self) -> None:
        self.assertEqual(recall_at_k(["d1", "d2"], {"d1", "d9"}), 0.5)

    def test_three_relevant_one_found_is_one_third(self) -> None:
        self.assertAlmostEqual(recall_at_k(["a", "x"], {"a", "b", "c"}), 1 / 3)

    def test_duplicate_chunks_count_once_for_recall(self) -> None:
        self.assertEqual(recall_at_k(["a", "a", "a", "b"], {"a", "b"}), 1.0)

    def test_duplicates_do_not_inflate_recall_above_one(self) -> None:
        self.assertEqual(recall_at_k(["a", "a", "a"], {"a"}), 1.0)

    def test_dedupe_happens_before_top_k_truncation(self) -> None:
        # a,a,a,b,c,d,e -> unique a,b,c,d,e -> e is unique rank 5 -> inside top 5.
        self.assertEqual(recall_at_k(["a", "a", "a", "b", "c", "d", "e"], {"e"}, k=5), 1.0)

    def test_empty_ranking_gives_zero(self) -> None:
        self.assertEqual(recall_at_k([], {"a"}), 0.0)

    def test_empty_relevant_set_raises_never_one(self) -> None:
        with self.assertRaises(ValidationError):
            recall_at_k(["a"], set())

    def test_nonpositive_k_raises(self) -> None:
        for k in (0, -1):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                recall_at_k(["a"], {"a"}, k=k)


class TestReciprocalRank(unittest.TestCase):
    """Per-question reciprocal rank."""

    def test_first_position_is_one(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], {"a"}), 1.0)

    def test_first_relevant_at_rank_two_is_half(self) -> None:
        self.assertEqual(reciprocal_rank(["p", "y", "x"], {"x", "y"}), 0.5)

    def test_rank_five_is_one_fifth(self) -> None:
        self.assertEqual(reciprocal_rank(["1", "2", "3", "4", "5", "6"], {"5"}), 0.2)

    def test_rank_six_outside_k_is_zero(self) -> None:
        self.assertEqual(reciprocal_rank(["1", "2", "3", "4", "5", "6"], {"6"}), 0.0)

    def test_duplicates_do_not_push_rank_down(self) -> None:
        # unique order: x, y -> y is rank 2, not rank 4.
        self.assertEqual(reciprocal_rank(["x", "x", "x", "y"], {"y"}), 0.5)

    def test_no_relevant_returns_zero(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], {"z"}), 0.0)

    def test_empty_relevant_raises(self) -> None:
        with self.assertRaises(ValidationError):
            reciprocal_rank(["a"], [])

    def test_nonpositive_k_raises(self) -> None:
        with self.assertRaises(ValidationError):
            reciprocal_rank(["a"], {"a"}, k=0)


class TestAverages(unittest.TestCase):
    """mean_recall, mrr, abstention helpers."""

    def test_mean_recall_is_simple_average(self) -> None:
        self.assertEqual(mean_recall([1.0, 0.5, 0.0]), 0.5)

    def test_mrr_of_one_half_zero_is_half(self) -> None:
        self.assertEqual(mrr([1.0, 0.5, 0.0]), 0.5)

    def test_empty_averages_raise(self) -> None:
        for fn in (mean_recall, mrr, abstention_accuracy):
            with self.subTest(fn=fn.__name__), self.assertRaises(ValidationError):
                fn([])

    def test_abstention_accuracy_fraction(self) -> None:
        self.assertAlmostEqual(abstention_accuracy([True, True, False]), 2 / 3)

    def test_abstains_when_no_results(self) -> None:
        self.assertTrue(abstention_correct(None, 0.5))

    def test_abstains_when_score_below_threshold(self) -> None:
        self.assertTrue(abstention_correct(0.49, 0.5))

    def test_score_equal_to_threshold_clears_it_so_not_abstention(self) -> None:
        self.assertFalse(abstention_correct(0.5, 0.5))

    def test_score_above_threshold_is_not_abstention(self) -> None:
        self.assertFalse(abstention_correct(0.9, 0.5))


SF, MH, UN = QuestionType.SINGLE_FACT, QuestionType.MULTI_HOP, QuestionType.UNANSWERABLE


def _q(qid: str, qtype: QuestionType, relevant: tuple[str, ...]) -> EvalQuestion:
    return EvalQuestion(
        question_id=qid,
        question=f"question {qid}",
        qtype=qtype,
        expected_answer="a",
        tenant_id="t1",
        relevant_doc_ids=relevant,
    )


def _r(
    qid: str, qtype: QuestionType, ranked: tuple[str, ...], score: float | None = None
) -> QuestionResult:
    return QuestionResult(qid, qtype, ranked, score)


class TestSummarize(unittest.TestCase):
    """Overall and per-type breakdown."""

    def setUp(self) -> None:
        self.questions = [
            _q("q1", SF, ("a",)),
            _q("q2", SF, ("b",)),
            _q("q3", MH, ("c", "d")),
            _q("q4", MH, ("e", "f")),
            _q("u1", UN, ()),
            _q("u2", UN, ()),
            _q("u3", UN, ()),
        ]
        self.results = [
            _r("q1", SF, ("a",)),  # recall 1, rr 1
            _r("q2", SF, ("x", "b")),  # recall 1, rr .5
            _r("q3", MH, ("z", "c")),  # recall .5, rr .5
            _r("q4", MH, ("z",)),  # recall 0, rr 0
            _r("u1", UN, ("a", "b"), 0.3),  # abstains (below .5)
            _r("u2", UN, ("a",), 0.5),  # equal to threshold -> does not abstain
            _r("u3", UN, (), None),  # abstains
        ]

    def test_overall_recall_excludes_unanswerable(self) -> None:
        report = summarize(self.questions, self.results, 0.5)
        self.assertEqual(report.recall, (1 + 1 + 0.5 + 0) / 4)

    def test_overall_mrr_excludes_unanswerable(self) -> None:
        report = summarize(self.questions, self.results, 0.5)
        self.assertEqual(report.mrr, (1 + 0.5 + 0.5 + 0) / 4)

    def test_counts(self) -> None:
        report = summarize(self.questions, self.results, 0.5)
        self.assertEqual((report.answerable_count, report.unanswerable_count, report.k), (4, 3, 5))

    def test_abstention_accuracy_scored_separately(self) -> None:
        report = summarize(self.questions, self.results, 0.5)
        self.assertAlmostEqual(report.abstention_accuracy or 0.0, 2 / 3)

    def test_per_type_breakdown_values_and_order(self) -> None:
        report = summarize(self.questions, self.results, 0.5)
        got = [(t.qtype, t.count, t.recall, t.mrr) for t in report.by_type]
        self.assertEqual(got, [(MH, 2, 0.25, 0.25), (SF, 2, 1.0, 0.75)])

    def test_input_order_does_not_change_report(self) -> None:
        forward = summarize(self.questions, self.results, 0.5)
        backward = summarize(self.questions[::-1], self.results[::-1], 0.5)
        self.assertEqual(forward, backward)

    def test_only_unanswerable_has_no_recall_or_mrr(self) -> None:
        report = summarize([_q("u1", UN, ())], [_r("u1", UN, (), None)], 0.5)
        self.assertIsNone(report.recall)
        self.assertIsNone(report.mrr)
        self.assertEqual(report.abstention_accuracy, 1.0)
        self.assertEqual(report.by_type, ())

    def test_only_answerable_has_no_abstention_accuracy(self) -> None:
        report = summarize([_q("q1", SF, ("a",))], [_r("q1", SF, ("a",))], 0.5)
        self.assertIsNone(report.abstention_accuracy)

    def test_k_is_applied(self) -> None:
        questions = [_q("q1", SF, ("c",))]
        results = [_r("q1", SF, ("a", "b", "c"))]
        self.assertEqual(summarize(questions, results, 0.5, k=2).recall, 0.0)
        self.assertEqual(summarize(questions, results, 0.5, k=3).recall, 1.0)

    def test_missing_result_raises(self) -> None:
        with self.assertRaisesRegex(ValidationError, "q4"):
            summarize(self.questions, [r for r in self.results if r.question_id != "q4"], 0.5)

    def test_duplicate_result_raises(self) -> None:
        with self.assertRaisesRegex(ValidationError, "q1"):
            summarize(self.questions, [*self.results, self.results[0]], 0.5)


def _yes(_: str) -> bool:
    return True


def _no(_: str) -> bool:
    return False


class TestCitationValid(unittest.TestCase):
    """Four conditions, first failure named, in spec order."""

    def _check(self, **overrides):
        lookups = {
            "exists": _yes,
            "authorized": _yes,
            "version_matches": _yes,
            "supports": _yes,
        }
        lookups.update(overrides)
        return citation_valid("doc-1", **lookups)

    def test_all_conditions_hold(self) -> None:
        result = self._check()
        self.assertTrue(result.valid)
        self.assertIsNone(result.failed)

    def test_each_single_failure_is_named(self) -> None:
        cases = {
            "exists": CitationCondition.EXISTS,
            "authorized": CitationCondition.AUTHORIZED,
            "version_matches": CitationCondition.VERSION_MATCHES,
            "supports": CitationCondition.SUPPORTS,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                result = self._check(**{name: _no})
                self.assertFalse(result.valid)
                self.assertEqual(result.failed, expected)

    def test_first_failure_wins_when_several_fail(self) -> None:
        result = self._check(authorized=_no, supports=_no)
        self.assertEqual(result.failed, CitationCondition.AUTHORIZED)

    def test_lookups_receive_the_citation(self) -> None:
        seen: list[str] = []

        def record(citation: str) -> bool:
            seen.append(citation)
            return True

        self._check(exists=record)
        self.assertEqual(seen, ["doc-1"])

    def test_empty_citation_raises(self) -> None:
        with self.assertRaises(ValidationError):
            citation_valid("  ", exists=_yes, authorized=_yes, version_matches=_yes, supports=_yes)


if __name__ == "__main__":
    unittest.main()
